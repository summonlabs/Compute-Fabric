#include "framework.h"

#include "compute_fabric/resource/reservation.h"
#include "compute_fabric/scheduler/scheduler.h"
#include "compute_fabric/state/state_locality.h"

using namespace cf;

namespace {
NodeRecord cpu_node(int slots) {
  NodeRecord n;
  n.id = Id128::random();
  n.session = Id128::random();
  n.epoch = 1;
  n.session_epoch = 1;
  n.health = NodeHealth::Online;
  n.connected = true;
  n.capacity.cpu_worker_slots = slots;
  n.capacity.host_memory_scheduling_bytes = 1 << 20;
  ExecutorCapability cpu;
  cpu.type = ExecutorType::Cpu;
  cpu.name = "cpu";
  cpu.slots = slots;
  cpu.healthy = true;
  n.capacity.executors.push_back(cpu);
  return n;
}
}  // namespace

TEST(reservation_acquire_release) {
  auto node = cpu_node(2);
  ReservationRegistry reg;
  auto r = reg.try_reserve(node, 1, Id128::random(), ExecutorType::Cpu, 1, 0);
  EXPECT_TRUE(r.reservation.has_value());
  EXPECT_EQ(reg.active_count(), 1ull);
  EXPECT_OK(reg.release(r.reservation->id));
  EXPECT_EQ(reg.active_count(), 0ull);
  EXPECT_FAIL(reg.release(r.reservation->id));  // double release
}

TEST(reservation_capacity_enforced) {
  auto node = cpu_node(2);
  ReservationRegistry reg;
  auto t1 = Id128::random();
  auto t2 = Id128::random();
  auto t3 = Id128::random();
  EXPECT_TRUE(reg.try_reserve(node, 1, t1, ExecutorType::Cpu, 1, 0).reservation);
  EXPECT_TRUE(reg.try_reserve(node, 1, t2, ExecutorType::Cpu, 1, 0).reservation);
  auto r3 = reg.try_reserve(node, 1, t3, ExecutorType::Cpu, 1, 0);
  EXPECT_FALSE(r3.reservation.has_value());
  EXPECT_FALSE(r3.reasons.empty());
  EXPECT_EQ(reg.active_count(), 2ull);
}

TEST(reservation_no_double_booking) {
  auto node = cpu_node(1);
  ReservationRegistry reg;
  auto t1 = Id128::random();
  auto t2 = Id128::random();
  EXPECT_TRUE(reg.try_reserve(node, 1, t1, ExecutorType::Cpu, 1, 0).reservation);
  EXPECT_FALSE(reg.try_reserve(node, 1, t2, ExecutorType::Cpu, 1, 0).reservation.has_value());
}

TEST(reservation_scratch_budget) {
  auto node = cpu_node(4);
  node.capacity.host_memory_scheduling_bytes = 100;
  ReservationRegistry reg;
  EXPECT_TRUE(
      reg.try_reserve(node, 1, Id128::random(), ExecutorType::Cpu, 1, 60)
          .reservation.has_value());
  EXPECT_FALSE(
      reg.try_reserve(node, 1, Id128::random(), ExecutorType::Cpu, 1, 60)
          .reservation.has_value());
}

TEST(reservation_stale_epoch_invalidated) {
  auto node = cpu_node(4);
  ReservationRegistry reg;
  auto r = reg.try_reserve(node, 7, Id128::random(), ExecutorType::Cpu, 1, 0);
  EXPECT_TRUE(r.reservation.has_value());
  size_t invalidated = reg.invalidate_before(8);
  EXPECT_EQ(invalidated, 1ull);
  EXPECT_EQ(reg.active_count(), 0ull);
}

TEST(reservation_epoch_bound_valid) {
  auto node = cpu_node(4);
  ReservationRegistry reg;
  auto r = reg.try_reserve(node, 8, Id128::random(), ExecutorType::Cpu, 1, 0);
  EXPECT_TRUE(r.reservation.has_value());
  size_t invalidated = reg.invalidate_before(8);
  EXPECT_EQ(invalidated, 0ull);
  EXPECT_EQ(reg.active_count(), 1ull);
}

TEST(reservation_offline_node_rejected) {
  auto node = cpu_node(4);
  node.health = NodeHealth::Offline;
  ReservationRegistry reg;
  auto r = reg.try_reserve(node, 1, Id128::random(), ExecutorType::Cpu, 1, 0);
  EXPECT_FALSE(r.reservation.has_value());
}

TEST(reservation_exclusive_slots_used_tracking) {
  auto node = cpu_node(4);
  ReservationRegistry reg;
  auto r1 = reg.try_reserve(node, 1, Id128::random(), ExecutorType::Cpu, 2, 0);
  EXPECT_TRUE(r1.reservation.has_value());
  EXPECT_EQ(reg.used_slots(node.id, ExecutorType::Cpu), 2u);
  auto r2 = reg.try_reserve(node, 1, Id128::random(), ExecutorType::Cpu, 2, 0);
  EXPECT_TRUE(r2.reservation.has_value());
  EXPECT_EQ(reg.used_slots(node.id, ExecutorType::Cpu), 4u);
  EXPECT_FALSE(reg.try_reserve(node, 1, Id128::random(), ExecutorType::Cpu, 1, 0)
                   .reservation.has_value());
}

TEST(reservation_session_invalidation) {
  auto node = cpu_node(4);
  ReservationRegistry reg;
  auto r = reg.try_reserve(node, 1, Id128::random(), ExecutorType::Cpu, 1, 0);
  EXPECT_TRUE(r.reservation.has_value());
  size_t n = reg.invalidate_session(node.id, node.session);
  EXPECT_EQ(n, 1ull);
  EXPECT_EQ(reg.active_count(), 0ull);
}

TEST(reservation_release_all_for_node) {
  auto node = cpu_node(4);
  ReservationRegistry reg;
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(reg.try_reserve(node, 1, Id128::random(), ExecutorType::Cpu, 1, 0)
                    .reservation.has_value());
  }
  reg.release_all_for_node(node.id);
  EXPECT_EQ(reg.active_count(), 0ull);
}
TEST(reservation_capacity_arithmetic_does_not_wrap) {
  auto node = cpu_node(2);
  ReservationRegistry reg;
  EXPECT_TRUE(reg.try_reserve(node, 1, Id128::random(), ExecutorType::Cpu, 1, 16).reservation);

  // 1 slot in use + 0xFFFFFFFF requested wraps to 0 in 32 bits and used to be
  // granted, over-committing the node.
  auto slots = reg.try_reserve(node, 1, Id128::random(), ExecutorType::Cpu, 0xFFFFFFFFu, 0);
  EXPECT_FALSE(slots.reservation.has_value());
  EXPECT_EQ(reg.used_slots(node.id, ExecutorType::Cpu), 1u);

  // 16 scratch bytes in use + (2^64 - 16) requested wraps to 0 in 64 bits.
  auto scratch = reg.try_reserve(node, 1, Id128::random(), ExecutorType::Cpu, 1,
                                 ~uint64_t{0} - 15);
  EXPECT_FALSE(scratch.reservation.has_value());
  EXPECT_EQ(reg.used_scratch(node.id), 16ull);
  EXPECT_EQ(reg.active_count(), 1ull);
}

TEST(scheduler_excludes_node_for_slot_request_above_int32) {
  auto node = cpu_node(2);
  std::map<NodeId, NodeRecord> nodes;
  nodes[node.id] = node;
  ComputeTask task;
  task.id = Id128::random();
  task.required_executor = ExecutorType::Cpu;
  task.required_cpu_slots = 0x80000000u;  // negative once narrowed to int32_t
  SchedulerPolicyEngine engine;
  ReservationRegistry reg;
  StateLocalityRegistry state;
  auto decision = engine.place(task, nodes, reg, state);
  EXPECT_FALSE(decision.feasible);
}
