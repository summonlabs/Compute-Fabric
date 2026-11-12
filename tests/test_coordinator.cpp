#include "framework.h"

#include <memory>
#include <thread>

#include "compute_fabric/coordinator/coordinator.h"
#include "compute_fabric/net/client.h"
#include "compute_fabric/node/node_runtime.h"

using namespace cf;

namespace {

std::string unique_store_dir() {
  return "test_store_coordinator_" + std::to_string(cf::Id128::random().lo());
}

ComputeTask mk_task(const char* name, uint64_t n = 10000) {
  ComputeTask t;
  t.id = Id128::random();
  t.workload = Id128::random();
  t.name = name;
  t.kernel = KernelType::HashInteger;
  t.kernel_param_n = n;
  t.kernel_param_a = 31;
  t.kernel_param_b = 17;
  t.seed = 42;
  t.estimated_duration_seconds = 0.01;
  return t;
}

TaskState wait_state(FabricClient& c, const ComputeTaskId& id,
                     const std::vector<TaskState>& targets, int64_t timeout_ms) {
  auto deadline = now_millis() + timeout_ms;
  while (now_millis() < deadline) {
    auto insp = c.inspect(id);
    if (insp.ok() && insp.value().found) {
      for (auto t : targets) {
        if (insp.value().state == t) return t;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  return TaskState::Submitted;
}

struct Fixture {
  std::string store_dir;
  std::unique_ptr<Coordinator> coord;
  uint16_t nport = 0;
  uint16_t cport = 0;
  std::vector<std::unique_ptr<NodeRuntime>> nodes;

  void start(const std::string& policy = "cost_aware") {
    if (store_dir.empty()) store_dir = unique_store_dir();
    CoordinatorConfig cfg;
    cfg.store_dir = store_dir;
    cfg.node_port = 0;
    cfg.client_port = 0;
    cfg.policy = SchedulerPolicy::CostAware;
    if (policy == "baseline") cfg.policy = SchedulerPolicy::Baseline;
    coord = std::make_unique<Coordinator>(cfg);
    auto ports = coord->start();
    EXPECT_TRUE(ports.ok());
    nport = ports.value().first;
    cport = ports.value().second;
  }

  void add_node(int cpu_slots = 2, bool cuda = false) {
    NodeConfig nc;
    nc.coordinator_host = "127.0.0.1";
    nc.coordinator_node_port = nport;
    nc.label = "node-" + std::to_string(nodes.size());
    nc.cpu_slots = cpu_slots;
    nc.enable_cuda = cuda;
    nc.cuda_device_ordinal = 0;
    nc.cuda_slots = cuda ? 1 : 0;
    auto node = std::make_unique<NodeRuntime>(nc);
    auto r = node->start();
    EXPECT_TRUE(r.ok());
    nodes.push_back(std::move(node));
  }

  std::unique_ptr<FabricClient> client() {
    auto c = std::make_unique<FabricClient>("127.0.0.1", cport);
    auto r = c->connect();
    EXPECT_TRUE(r.ok());
    return c;
  }

  void shutdown() {
    for (auto& n : nodes) {
      if (n) n->stop();
    }
    nodes.clear();
    if (coord) coord->stop();
  }

  ~Fixture() { shutdown(); }
};

}  // namespace

TEST(coordinator_full_task_flow) {
  Fixture f;
  f.start();
  f.add_node(2);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(1, 10000));

  auto task = mk_task("flow");
  auto r = c->submit(task.workload, "w", task);
  EXPECT_TRUE(r.ok());
  EXPECT_TRUE(r.value().accepted);

  auto st = wait_state(*c, task.id, {TaskState::Succeeded, TaskState::Failed},
                       15000);
  EXPECT_EQ(st, TaskState::Succeeded);

  auto insp = c->inspect(task.id);
  EXPECT_TRUE(insp.ok());
  EXPECT_EQ(insp.value().state, TaskState::Succeeded);
  EXPECT_EQ(insp.value().attempts.size(), 1ull);
  EXPECT_FALSE(insp.value().attempts[0].output_integrity.empty());
  EXPECT_FALSE(insp.value().attempts[0].node_id.is_zero());
}

TEST(coordinator_illegal_dependency_rejected) {
  Fixture f;
  f.start();
  f.add_node(2);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(1, 10000));

  auto task = mk_task("missing-dep");
  task.dependencies = {Id128::random()};
  auto r = c->submit(task.workload, "w", task);
  EXPECT_TRUE(r.ok());
  EXPECT_FALSE(r.value().accepted);
  EXPECT_STR_CONTAINS(r.value().error, "missing dependency");

  // Self-dependency.
  auto self = mk_task("self");
  self.dependencies = {self.id};
  auto r2 = c->submit(self.workload, "w", self);
  EXPECT_TRUE(r2.ok());
  EXPECT_FALSE(r2.value().accepted);
}

TEST(coordinator_dag_execution_order) {
  Fixture f;
  f.start();
  f.add_node(2);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(1, 10000));

  auto a = mk_task("a");
  auto b = mk_task("b");
  auto d = mk_task("d");
  b.dependencies = {a.id};
  d.dependencies = {b.id};

  EXPECT_TRUE(c->submit(a.workload, "w", a).value().accepted);
  EXPECT_TRUE(c->submit(b.workload, "w", b).value().accepted);
  EXPECT_TRUE(c->submit(d.workload, "w", d).value().accepted);

  auto st_b = wait_state(*c, b.id, {TaskState::Succeeded, TaskState::Failed}, 15000);
  EXPECT_EQ(st_b, TaskState::Succeeded);
  // d only becomes runnable once b has succeeded, so it finishes after b is
  // first observed as Succeeded: wait for it before checking its state.
  auto st_d = wait_state(*c, d.id, {TaskState::Succeeded, TaskState::Failed}, 15000);
  EXPECT_EQ(st_d, TaskState::Succeeded);
  // d must not have succeeded before b.
  auto insp_b = c->inspect(b.id);
  EXPECT_EQ(c->inspect(d.id).value().state, TaskState::Succeeded);
  EXPECT_TRUE(insp_b.ok());
}

TEST(coordinator_retry_on_node_loss) {
  Fixture f;
  f.start();
  f.add_node(2);
  f.add_node(2);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(2, 10000));

  auto task = mk_task("retry", 20000000);
  task.kernel = KernelType::SyntheticLoop;
  task.retry.max_attempts = 3;
  task.retry.retry_on_node_loss = true;
  task.retry.retry_backoff_ms = 50;
  task.estimated_duration_seconds = 1.0;
  EXPECT_TRUE(c->submit(task.workload, "w", task).value().accepted);

  // Wait until the task is Running, then kill the node that hosts it.
  auto deadline = now_millis() + 10000;
  bool running = false;
  NodeId host;
  ExecutionAttempt first;
  while (now_millis() < deadline) {
    auto insp = c->inspect(task.id);
    if (insp.ok() && insp.value().state == TaskState::Running &&
        !insp.value().attempts.empty()) {
      host = insp.value().attempts.back().node_id;
      first = insp.value().attempts.back();
      running = true;
      break;
    }
    if (insp.ok() && is_terminal(insp.value().state)) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(running);
  if (running) {
    for (auto& n : f.nodes) {
      if (n && n->node_id() == host) {
        n->stop();
        n.reset();
      }
    }
    auto st = wait_state(*c, task.id, {TaskState::Succeeded, TaskState::Failed},
                         25000);
    EXPECT_EQ(st, TaskState::Succeeded);
    auto insp = c->inspect(task.id);
    EXPECT_GE(insp.value().attempts.size(), 1ull);
  }
}

TEST(coordinator_stale_epoch_rejects_old_completion) {
  Fixture f;
  f.start();
  f.add_node(2);
  f.add_node(2);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(2, 10000));

  // A task that will be lost on its first node and retried on the second.
  // Keep the first attempt observably active long enough to stop its node.
  // HashInteger can finish between the 25 ms inspection polls in Release
  // builds, which would test polling speed instead of stale-attempt rejection.
  auto task = mk_task("stale", 20000000);
  task.kernel = KernelType::SyntheticLoop;
  task.estimated_duration_seconds = 1.0;
  task.retry.max_attempts = 3;
  task.retry.retry_on_node_loss = true;
  task.retry.retry_backoff_ms = 50;
  EXPECT_TRUE(c->submit(task.workload, "w", task).value().accepted);

  // Wait for the first attempt to start, capture its token, then kill its node.
  auto deadline = now_millis() + 10000;
  ExecutionAttempt first_attempt;
  bool captured = false;
  while (now_millis() < deadline) {
    auto insp = c->inspect(task.id);
    if (insp.ok() && !insp.value().attempts.empty() &&
        insp.value().attempts[0].state == AttemptState::Started) {
      first_attempt = insp.value().attempts[0];
      captured = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  EXPECT_TRUE(captured);
  if (!captured) return;
  for (auto& n : f.nodes) {
    if (n && n->node_id() == first_attempt.node_id) {
      n->stop();
      n.reset();
    }
  }
  // Wait for the retry to succeed on the surviving node.
  auto st = wait_state(*c, task.id, {TaskState::Succeeded, TaskState::Failed},
                       25000);
  EXPECT_EQ(st, TaskState::Succeeded);
  auto final_insp = c->inspect(task.id);
  EXPECT_GE(final_insp.value().attempts.size(), 2ull);

  // Simulate the dead node waking up and reporting the OLD (superseded)
  // attempt. The coordinator must reject it.
  auto sock = net::TcpSocket::connect("127.0.0.1", f.nport, 5000);
  EXPECT_TRUE(sock.ok());
  net::Channel ch(sock.take());
  proto::Hello hello;
  hello.role = "node";
  proto::BinaryWriter hw;
  proto::serialize(hw, hello);
  EXPECT_OK(ch.send_message(proto::kHello, proto::SenderKind::Node,
                            Id128::random(), 1, hw.bytes(), 5000));
  proto::NodeRegister reg;
  reg.node_id = Id128::random();
  reg.label = "fake";
  NodeCapacity cap;
  cap.cpu_worker_slots = 1;
  ExecutorCapability cpu;
  cpu.type = ExecutorType::Cpu;
  cpu.name = "cpu";
  cpu.slots = 1;
  cpu.healthy = true;
  cap.executors.push_back(cpu);
  reg.capacity = cap;
  proto::BinaryWriter rw;
  proto::serialize(rw, reg);
  EXPECT_OK(ch.send_message(proto::kNodeRegister, proto::SenderKind::Node,
                            reg.node_id, 2, rw.bytes(), 5000));
  deadline = now_millis() + 10000;
  bool acked = false;
  while (now_millis() < deadline) {
    auto frame = ch.recv_frame(500, proto::kMaxFramePayload);
    if (frame.ok() &&
        frame.value().header.message_type == proto::kNodeRegisterAck) {
      acked = true;
      break;
    }
  }
  EXPECT_TRUE(acked);

  proto::AttemptComplete co;
  co.attempt_id = first_attempt.id;
  co.task_id = first_attempt.task_id;
  co.reservation_id = first_attempt.reservation_id;
  co.coordinator_epoch = first_attempt.coordinator_epoch;
  co.result_code = 0;
  co.output_integrity = "superseded";
  proto::BinaryWriter w;
  proto::serialize(w, co);
  EXPECT_OK(ch.send_message(proto::kAttemptComplete, proto::SenderKind::Node,
                            reg.node_id, 3, w.bytes(), 5000));
  auto resp = ch.recv_frame(5000, proto::kMaxFramePayload);
  EXPECT_TRUE(resp.ok());
  if (resp.ok() && resp.value().header.message_type == proto::kAttemptCompleteAck) {
    proto::BinaryReader br(resp.value().payload);
    auto ackr = proto::deserialize_attempt_ack(br);
    EXPECT_TRUE(ackr.ok());
    EXPECT_FALSE(ackr.value().accepted);
  }
  ch.close();
}

TEST(coordinator_idempotent_duplicate_submit) {
  Fixture f;
  f.start();
  f.add_node(2);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(1, 10000));

  auto task = mk_task("dup");
  auto r1 = c->submit(task.workload, "w", task);
  auto r2 = c->submit(task.workload, "w", task);
  EXPECT_TRUE(r1.ok() && r2.ok());
  EXPECT_TRUE(r1.value().accepted);
  EXPECT_TRUE(r2.value().accepted);
  EXPECT_EQ(r1.value().task_id, r2.value().task_id);
  auto list = c->list("");
  EXPECT_TRUE(list.ok());
  int count = 0;
  for (const auto& e : list.value().tasks) {
    if (e.task_id == task.id) ++count;
  }
  EXPECT_EQ(count, 1);
}

TEST(coordinator_idempotent_duplicate_cancel) {
  Fixture f;
  f.start();
  f.add_node(2);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(1, 10000));

  auto task = mk_task("cancel-me");
  EXPECT_TRUE(c->submit(task.workload, "w", task).value().accepted);
  auto r1 = c->cancel(task.id);
  auto r2 = c->cancel(task.id);
  EXPECT_TRUE(r1.ok() && r2.ok());
  (void)r1;
  (void)r2;
  // A task that was already running is cancelled cooperatively: it turns
  // terminal when its node reports back, not when the cancel request returns.
  auto deadline = now_millis() + 10000;
  auto insp = c->inspect(task.id);
  while (now_millis() < deadline &&
         !(insp.ok() && is_terminal(insp.value().state))) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    insp = c->inspect(task.id);
  }
  EXPECT_TRUE(insp.ok());
  EXPECT_TRUE(is_terminal(insp.value().state));
}

TEST(coordinator_cancel_queued_task) {
  Fixture f;
  f.start();
  f.add_node(1);  // single slot
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(1, 10000));

  // Fill the single slot with a long-running blocker.
  auto blocker = mk_task("blocker", 64000000);
  blocker.kernel = KernelType::SyntheticLoop;
  blocker.retry.max_attempts = 1;
  EXPECT_TRUE(c->submit(blocker.workload, "w", blocker).value().accepted);
  auto victim = mk_task("victim", 1000);
  victim.retry.max_attempts = 1;
  EXPECT_TRUE(c->submit(victim.workload, "w", victim).value().accepted);

  // Wait until the coordinator has accepted both (victim non-terminal), then
  // cancel the queued victim immediately while the blocker holds the slot.
  auto deadline = now_millis() + 5000;
  while (now_millis() < deadline) {
    auto insp = c->inspect(victim.id);
    if (insp.ok() && insp.value().found && !is_terminal(insp.value().state)) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  auto rc = c->cancel(victim.id);
  EXPECT_TRUE(rc.ok());
  EXPECT_TRUE(rc.value().found);
  EXPECT_TRUE(rc.value().cancelled);
  auto insp = c->inspect(victim.id);
  EXPECT_TRUE(insp.ok());
  EXPECT_EQ(insp.value().state, TaskState::Cancelled);
  // Blocker still completes.
  auto st = wait_state(*c, blocker.id, {TaskState::Succeeded, TaskState::Failed}, 30000);
  EXPECT_EQ(st, TaskState::Succeeded);
}

TEST(coordinator_node_drain) {
  Fixture f;
  f.start();
  f.add_node(2);
  f.add_node(2);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(2, 10000));
  auto nodes = c->nodes();
  EXPECT_TRUE(nodes.ok());
  EXPECT_EQ(nodes.value().nodes.size(), 2ull);
  auto nid = nodes.value().nodes[0].node_id;
  auto dr = c->drain(nid, true);
  EXPECT_TRUE(dr.ok());
  EXPECT_TRUE(dr.value().accepted);
  auto nodes2 = c->nodes();
  int drained = 0;
  for (const auto& e : nodes2.value().nodes) {
    if (e.health == NodeHealth::Draining) ++drained;
  }
  EXPECT_EQ(drained, 1);
  auto udr = c->drain(nid, false);
  EXPECT_TRUE(udr.ok());
  EXPECT_TRUE(udr.value().accepted);
}

TEST(coordinator_concurrent_submissions) {
  Fixture f;
  f.start();
  f.add_node(4);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(1, 10000));

  std::vector<std::thread> threads;
  std::vector<ComputeTaskId> ids;
  std::mutex mu;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&, i] {
      auto cc = f.client();
      auto task = mk_task(("conc-" + std::to_string(i)).c_str(), 50000);
      auto r = cc->submit(task.workload, "w", task);
      if (r.ok() && r.value().accepted) {
        std::lock_guard<std::mutex> lk(mu);
        ids.push_back(task.id);
      }
    });
  }
  for (auto& t : threads) t.join();
  EXPECT_EQ(ids.size(), 8ull);

  auto deadline = now_millis() + 30000;
  int succeeded = 0;
  while (now_millis() < deadline) {
    succeeded = 0;
    for (const auto& id : ids) {
      auto insp = c->inspect(id);
      if (insp.ok() && insp.value().state == TaskState::Succeeded) ++succeeded;
    }
    if (succeeded == 8) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  EXPECT_EQ(succeeded, 8);
}

TEST(coordinator_shutdown_during_active_attempt) {
  Fixture f;
  f.start();
  f.add_node(2);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(1, 10000));
  auto task = mk_task("long", 5000000);
  EXPECT_TRUE(c->submit(task.workload, "w", task).value().accepted);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  // Shutdown coordinator while the attempt is running.
  auto s = c->shutdown_coordinator();
  EXPECT_TRUE(s.ok());
  f.coord->stop();
  // Node should still be stopped cleanly.
  f.nodes[0]->stop();
  EXPECT_TRUE(true);
}

TEST(coordinator_epoch_bump_on_restart) {
  Fixture f;
  f.start();
  auto e1 = f.coord->epoch();
  f.coord->stop();
  f.coord.reset();
  f.start();
  auto e2 = f.coord->epoch();
  EXPECT_GT(e2, e1);
}

TEST(coordinator_persistence_recovery) {
  std::string dir = unique_store_dir();
  {
    CoordinatorConfig cfg;
    cfg.store_dir = dir;
    cfg.node_port = 0;
    cfg.client_port = 0;
    Coordinator coord(cfg);
    auto ports = coord.start();
    EXPECT_TRUE(ports.ok());
    NodeConfig nc;
    nc.coordinator_node_port = ports.value().first;
    nc.cpu_slots = 2;
    NodeRuntime node(nc);
    EXPECT_OK(node.start());
    FabricClient c("127.0.0.1", ports.value().second);
    EXPECT_OK(c.connect());
    auto task = mk_task("persist", 30000);
    EXPECT_TRUE(c.submit(task.workload, "w", task).value().accepted);
    auto st = wait_state(c, task.id, {TaskState::Succeeded}, 15000);
    EXPECT_EQ(st, TaskState::Succeeded);
    node.stop();
    c.close();
    coord.stop();
  }
  {
    // Restart with the same store dir: the committed task must be retained.
    CoordinatorConfig cfg;
    cfg.store_dir = dir;
    cfg.node_port = 0;
    cfg.client_port = 0;
    Coordinator coord(cfg);
    auto ports = coord.start();
    EXPECT_TRUE(ports.ok());
    FabricClient c("127.0.0.1", ports.value().second);
    EXPECT_OK(c.connect());
    auto list = c.list("");
    EXPECT_TRUE(list.ok());
    bool found_committed = false;
    for (const auto& e : list.value().tasks) {
      if (e.state == TaskState::Succeeded) found_committed = true;
    }
    EXPECT_TRUE(found_committed);
    c.shutdown_coordinator();
    c.close();
    coord.stop();
  }
}

TEST(coordinator_heartbeat_and_node_list) {
  Fixture f;
  f.start();
  f.add_node(2);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(1, 10000));
  auto st = c->status();
  EXPECT_TRUE(st.ok());
  EXPECT_EQ(st.value().node_count, 1u);
  EXPECT_EQ(st.value().coordinator_epoch, f.coord->epoch());
}

TEST(coordinator_placement_query) {
  Fixture f;
  f.start();
  f.add_node(2);
  f.add_node(2);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(2, 10000));
  auto task = mk_task("query");
  task.required_executor = ExecutorType::Cpu;
  auto r = c->placement_query(task);
  EXPECT_TRUE(r.ok());
  EXPECT_TRUE(r.value().found);
  EXPECT_TRUE(r.value().decision.feasible);
  EXPECT_TRUE(r.value().decision.chosen_node.has_value());
  EXPECT_FALSE(r.value().decision.candidates.empty());
}

TEST(coordinator_idempotency_cache_dedup) {
  IdempotencyCache cache;
  std::string key = "node:1";
  uint16_t mt = 0;
  std::vector<uint8_t> payload;
  EXPECT_FALSE(cache.get(key, mt, payload));
  cache.put(key, proto::kTaskSubmitResponse, {1, 2, 3});
  EXPECT_TRUE(cache.get(key, mt, payload));
  EXPECT_EQ(mt, proto::kTaskSubmitResponse);
  EXPECT_EQ(payload.size(), 3ull);
  // Re-put does not overwrite.
  cache.put(key, proto::kTaskCancelResponse, {9});
  uint16_t mt2 = 0;
  std::vector<uint8_t> payload2;
  EXPECT_TRUE(cache.get(key, mt2, payload2));
  EXPECT_EQ(mt2, proto::kTaskSubmitResponse);
}

TEST(coordinator_cuda_placement_via_fabric) {
  // Skip entirely when this build has no CUDA device available.
  if (enumerate_cuda_devices().empty()) {
    EXPECT_TRUE(true);
    return;
  }
  Fixture f;
  f.start();
  f.add_node(2, /*cuda=*/true);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(1, 10000));

  auto task = mk_task("cuda-task");
  task.required_executor = ExecutorType::Cuda;
  task.kernel = KernelType::VectorTransform;
  task.kernel_param_n = 100000;
  auto r = c->submit(task.workload, "w", task);
  EXPECT_TRUE(r.ok() && r.value().accepted);
  auto st = wait_state(*c, task.id, {TaskState::Succeeded, TaskState::Failed},
                       20000);
  EXPECT_EQ(st, TaskState::Succeeded);
  auto insp = c->inspect(task.id);
  EXPECT_TRUE(insp.ok());
  if (!insp.value().attempts.empty()) {
    EXPECT_STR_CONTAINS(insp.value().attempts[0].device, "cuda");
  }
}

// ---------------------------------------------------------------------------
// Authority fencing and retry accounting. These tests drive the coordinator
// with scripted node connections so every interleaving is deterministic.
// ---------------------------------------------------------------------------
namespace {

// A node connection scripted by the test: it registers, receives dispatches,
// and reports whatever the test tells it to. It never heartbeats.
struct ScriptedNode {
  std::unique_ptr<net::Channel> ch;
  NodeId id = Id128::random();
  uint64_t request = 10;

  bool join(uint16_t node_port) {
    auto sock = net::TcpSocket::connect("127.0.0.1", node_port, 5000);
    if (sock.failed()) return false;
    ch = std::make_unique<net::Channel>(sock.take());
    proto::Hello hello;
    hello.role = "node";
    proto::BinaryWriter hw;
    proto::serialize(hw, hello);
    if (ch->send_message(proto::kHello, proto::SenderKind::Node, id, 1,
                         hw.bytes(), 5000).failed()) {
      return false;
    }
    proto::NodeRegister reg;
    reg.node_id = id;
    reg.label = "scripted";
    NodeCapacity cap;
    cap.cpu_worker_slots = 1;
    ExecutorCapability cpu;
    cpu.type = ExecutorType::Cpu;
    cpu.name = "cpu";
    cpu.slots = 1;
    cpu.healthy = true;
    cap.executors.push_back(cpu);
    reg.capacity = cap;
    proto::BinaryWriter rw;
    proto::serialize(rw, reg);
    if (ch->send_message(proto::kNodeRegister, proto::SenderKind::Node, id, 2,
                         rw.bytes(), 5000).failed()) {
      return false;
    }
    return wait_for(proto::kNodeRegisterAck, nullptr, 10000);
  }

  bool wait_for(uint16_t type, std::vector<uint8_t>* payload,
                int64_t timeout_ms) {
    const int64_t deadline = now_millis() + timeout_ms;
    while (now_millis() < deadline) {
      auto frame = ch->recv_frame(250, proto::kMaxFramePayload);
      if (frame.ok() && frame.value().header.message_type == type) {
        if (payload) *payload = frame.value().payload;
        return true;
      }
    }
    return false;
  }

  bool next_dispatch(proto::DispatchAttempt& out, int64_t timeout_ms = 10000) {
    std::vector<uint8_t> payload;
    if (!wait_for(proto::kDispatchAttempt, &payload, timeout_ms)) return false;
    proto::BinaryReader r(payload);
    auto m = proto::deserialize_dispatch_attempt(r);
    if (m.failed()) return false;
    out = m.value();
    return true;
  }

  void started(const proto::DispatchAttempt& d) {
    proto::AttemptStarted st;
    st.attempt_id = d.attempt_id;
    st.started_at_ms = now_millis();
    st.device = "cpu";
    proto::BinaryWriter w;
    proto::serialize(w, st);
    (void)ch->send_message(proto::kAttemptStarted, proto::SenderKind::Node, id,
                           ++request, w.bytes(), 5000);
  }

  // Each report returns whether the coordinator accepted it.
  bool complete(const proto::DispatchAttempt& d, const std::string& integrity) {
    proto::AttemptComplete co;
    co.attempt_id = d.attempt_id;
    co.task_id = d.task.id;
    co.reservation_id = d.reservation_id;
    co.coordinator_epoch = d.coordinator_epoch;
    co.output_integrity = integrity;
    proto::BinaryWriter w;
    proto::serialize(w, co);
    (void)ch->send_message(proto::kAttemptComplete, proto::SenderKind::Node, id,
                           ++request, w.bytes(), 5000);
    return accepted(proto::kAttemptCompleteAck);
  }

  bool failed(const proto::DispatchAttempt& d) {
    proto::AttemptFailed fa;
    fa.attempt_id = d.attempt_id;
    fa.task_id = d.task.id;
    fa.reservation_id = d.reservation_id;
    fa.coordinator_epoch = d.coordinator_epoch;
    fa.failure_kind = static_cast<uint8_t>(FailureKind::ExecutorFailed);
    fa.failure_reason = "scripted failure";
    proto::BinaryWriter w;
    proto::serialize(w, fa);
    (void)ch->send_message(proto::kAttemptFailed, proto::SenderKind::Node, id,
                           ++request, w.bytes(), 5000);
    return accepted(proto::kAttemptFailedAck);
  }

  bool accepted(uint16_t ack_type) {
    std::vector<uint8_t> payload;
    if (!wait_for(ack_type, &payload, 5000)) return false;
    proto::BinaryReader r(payload);
    auto ack = proto::deserialize_attempt_ack(r);
    return ack.ok() && ack.value().accepted;
  }

  void leave() {
    if (ch) ch->close();
  }
};

// A coordinator whose nodes are only ever lost by an explicit disconnect.
struct ScriptedFabric {
  std::string store_dir = unique_store_dir();
  std::unique_ptr<Coordinator> coord;
  uint16_t nport = 0;
  uint16_t cport = 0;

  void start() {
    CoordinatorConfig cfg;
    cfg.store_dir = store_dir;
    cfg.node_port = 0;
    cfg.client_port = 0;
    cfg.heartbeat_timeout_ms = 60000;  // scripted nodes never heartbeat
    coord = std::make_unique<Coordinator>(cfg);
    auto ports = coord->start();
    EXPECT_TRUE(ports.ok());
    nport = ports.value().first;
    cport = ports.value().second;
  }

  std::unique_ptr<FabricClient> client() {
    auto c = std::make_unique<FabricClient>("127.0.0.1", cport);
    EXPECT_TRUE(c->connect().ok());
    return c;
  }

  ~ScriptedFabric() {
    if (coord) coord->stop();
  }
};

ComputeTask retryable_task(const char* name, uint32_t max_attempts) {
  auto task = mk_task(name);
  task.retry.max_attempts = max_attempts;
  task.retry.retry_on_node_loss = true;
  task.retry.retry_on_executor_failure = true;
  task.retry.retry_backoff_ms = 10;
  return task;
}

}  // namespace

TEST(coordinator_superseded_attempt_cannot_touch_task_owned_by_retry) {
  ScriptedFabric f;
  f.start();
  auto c = f.client();
  auto task = retryable_task("superseded", 3);

  // Attempt 0 starts on the first node, which is then lost.
  ScriptedNode first;
  EXPECT_TRUE(first.join(f.nport));
  EXPECT_TRUE(c->submit(task.workload, "w", task).value().accepted);
  proto::DispatchAttempt old_attempt;
  EXPECT_TRUE(first.next_dispatch(old_attempt));
  first.started(old_attempt);
  EXPECT_EQ(wait_state(*c, task.id, {TaskState::Running}, 5000), TaskState::Running);
  first.leave();

  // The retry (attempt 1) is now running on a second node.
  ScriptedNode second;
  EXPECT_TRUE(second.join(f.nport));
  proto::DispatchAttempt retry;
  EXPECT_TRUE(second.next_dispatch(retry));
  EXPECT_NE(retry.attempt_id, old_attempt.attempt_id);
  second.started(retry);
  auto deadline = now_millis() + 5000;
  bool retry_running = false;
  while (now_millis() < deadline && !retry_running) {
    auto insp = c->inspect(task.id);
    retry_running = insp.ok() && insp.value().state == TaskState::Running &&
                    insp.value().attempts.size() == 2 &&
                    insp.value().attempts[1].state == AttemptState::Started;
    if (!retry_running) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(retry_running);

  // The lost node wakes up and reports for the superseded attempt. Neither a
  // failure nor a completion may be accepted, and neither may disturb the
  // task that the retry now owns.
  ScriptedNode waking;
  EXPECT_TRUE(waking.join(f.nport));
  EXPECT_FALSE(waking.failed(old_attempt));
  EXPECT_EQ(c->inspect(task.id).value().state, TaskState::Running);
  EXPECT_FALSE(waking.complete(old_attempt, "superseded"));
  EXPECT_EQ(c->inspect(task.id).value().state, TaskState::Running);

  // The retry still commits, and it is the retry's result that is recorded.
  EXPECT_TRUE(second.complete(retry, "current"));
  EXPECT_EQ(wait_state(*c, task.id, {TaskState::Succeeded, TaskState::Failed}, 5000),
            TaskState::Succeeded);
  auto final_insp = c->inspect(task.id);
  EXPECT_TRUE(final_insp.ok());
  int committed = 0;
  for (const auto& a : final_insp.value().attempts) {
    if (!a.committed) continue;
    ++committed;
    EXPECT_EQ(a.id, retry.attempt_id);
    EXPECT_EQ(a.output_integrity, std::string("current"));
  }
  EXPECT_EQ(committed, 1);
  waking.leave();
  second.leave();
}

TEST(coordinator_node_loss_retry_respects_max_attempts) {
  ScriptedFabric f;
  f.start();
  auto c = f.client();
  auto task = retryable_task("bounded", 2);

  ScriptedNode first;
  EXPECT_TRUE(first.join(f.nport));
  EXPECT_TRUE(c->submit(task.workload, "w", task).value().accepted);
  proto::DispatchAttempt a0;
  EXPECT_TRUE(first.next_dispatch(a0));
  EXPECT_EQ(a0.attempt_number, 0u);
  first.started(a0);
  EXPECT_EQ(wait_state(*c, task.id, {TaskState::Running}, 5000), TaskState::Running);
  first.leave();

  ScriptedNode second;
  EXPECT_TRUE(second.join(f.nport));
  proto::DispatchAttempt a1;
  EXPECT_TRUE(second.next_dispatch(a1));
  EXPECT_EQ(a1.attempt_number, 1u);
  second.started(a1);
  second.leave();

  // Both permitted attempts were lost: the task fails instead of being
  // requeued for a third attempt.
  EXPECT_EQ(wait_state(*c, task.id, {TaskState::Failed}, 5000), TaskState::Failed);
  EXPECT_EQ(c->inspect(task.id).value().attempts.size(), 2ull);
}

TEST(coordinator_restart_keeps_the_retry_budget) {
  ScriptedFabric f;
  f.start();
  auto task = retryable_task("durable-budget", 2);
  {
    auto c = f.client();
    ScriptedNode first;
    EXPECT_TRUE(first.join(f.nport));
    EXPECT_TRUE(c->submit(task.workload, "w", task).value().accepted);
    proto::DispatchAttempt a0;
    EXPECT_TRUE(first.next_dispatch(a0));
    first.started(a0);
    EXPECT_EQ(wait_state(*c, task.id, {TaskState::Running}, 5000), TaskState::Running);
    c->close();
    f.coord->stop();
    f.coord.reset();
    first.leave();
  }

  // Restart on the same store. The interrupted attempt already used one of
  // the two permitted attempts, so the next one is attempt 1 and is the last.
  f.start();
  auto c = f.client();
  ScriptedNode second;
  EXPECT_TRUE(second.join(f.nport));
  proto::DispatchAttempt a1;
  EXPECT_TRUE(second.next_dispatch(a1));
  EXPECT_EQ(a1.attempt_number, 1u);
  second.started(a1);
  EXPECT_EQ(wait_state(*c, task.id, {TaskState::Running}, 5000), TaskState::Running);
  second.leave();
  EXPECT_EQ(wait_state(*c, task.id, {TaskState::Failed}, 5000), TaskState::Failed);
}

TEST(node_runtime_stop_wakes_every_run_that_is_entering_its_wait) {
  // run() waits without a timeout for the stop flag. A stop request that
  // notifies without passing through the wait's mutex is lost when it lands
  // between a run() testing the flag and blocking, and that run() then never
  // returns. Several threads enter run() back to back while stop() fires, so
  // one of them is inside that window when the flag is set.
  const unsigned cores = std::thread::hardware_concurrency();
  const int runners = cores >= 8 ? 6 : 2;
  constexpr int kCycles = 3000;

  std::atomic<NodeRuntime*> target{nullptr};
  std::atomic<int> round{0};
  std::atomic<int> returned{0};
  std::atomic<bool> quit{false};

  std::vector<std::thread> pool;
  for (int i = 0; i < runners; ++i) {
    pool.emplace_back([&] {
      int seen = 0;
      for (;;) {
        // Spin at the gate so every runner is released at the same instant.
        unsigned spins = 0;
        while (round.load(std::memory_order_acquire) == seen) {
          if (quit.load(std::memory_order_acquire)) return;
          if ((++spins & 0xFFFu) == 0) std::this_thread::yield();
        }
        ++seen;
        target.load(std::memory_order_acquire)->run();
        returned.fetch_add(1, std::memory_order_release);
      }
    });
  }

  int completed = 0;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    NodeConfig nc;
    nc.cpu_slots = 1;
    NodeRuntime node(nc);
    target.store(&node, std::memory_order_release);
    returned.store(0, std::memory_order_release);
    round.fetch_add(1, std::memory_order_release);
    const auto gap_end = std::chrono::steady_clock::now() +
                         std::chrono::nanoseconds((cycle % 64) * 100);
    while (std::chrono::steady_clock::now() < gap_end) {
    }
    node.stop();
    // Every runner has to come back before the node can be destroyed.
    while (returned.load(std::memory_order_acquire) < runners) {
      std::this_thread::yield();
    }
    ++completed;
  }
  quit.store(true, std::memory_order_release);
  for (auto& t : pool) t.join();
  EXPECT_EQ(completed, kCycles);
}

// ---------------------------------------------------------------------------
// Lifecycle: reports that arrive before a start, and blocked tasks.
// ---------------------------------------------------------------------------
TEST(coordinator_attempt_rejected_before_start_fails_the_task) {
  Fixture f;
  f.start();
  f.add_node(2);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(1, 10000));

  // No executor supports this kernel, so the node rejects the dispatch: it
  // reports a failure without ever reporting a start, and the task is still
  // Dispatching when that failure arrives.
  auto task = mk_task("no-kernel");
  task.kernel = KernelType::None;
  EXPECT_TRUE(c->submit(task.workload, "w", task).value().accepted);

  // One attempt is allowed. The task fails; it is not dispatched again.
  EXPECT_EQ(wait_state(*c, task.id, {TaskState::Failed, TaskState::Succeeded}, 4000),
            TaskState::Failed);
  auto insp = c->inspect(task.id);
  EXPECT_TRUE(insp.ok());
  EXPECT_EQ(insp.value().attempts.size(), 1ull);
}

TEST(coordinator_completion_without_start_report_commits_the_task) {
  ScriptedFabric f;
  f.start();
  auto c = f.client();
  auto task = retryable_task("unstarted", 1);

  ScriptedNode node;
  EXPECT_TRUE(node.join(f.nport));
  EXPECT_TRUE(c->submit(task.workload, "w", task).value().accepted);
  proto::DispatchAttempt d;
  EXPECT_TRUE(node.next_dispatch(d));

  // The coordinator accepts a completion for an attempt that never reported a
  // start. Having accepted it, the task must be Succeeded, not left
  // Dispatching to be requeued and executed a second time.
  EXPECT_TRUE(node.complete(d, "result"));
  auto insp = c->inspect(task.id);
  EXPECT_TRUE(insp.ok());
  EXPECT_EQ(insp.value().state, TaskState::Succeeded);
  EXPECT_EQ(insp.value().attempts.size(), 1ull);
  node.leave();
}

TEST(coordinator_dependents_of_a_failed_task_fail) {
  Fixture f;
  f.start();
  f.add_node(2);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(1, 10000));

  // The upstream task fails while running: its size is over the kernel bound.
  auto upstream = mk_task("upstream", 65u * 1000u * 1000u);
  auto child = mk_task("child");
  auto grandchild = mk_task("grandchild");
  child.dependencies = {upstream.id};
  grandchild.dependencies = {child.id};
  EXPECT_TRUE(c->submit(upstream.workload, "w", upstream).value().accepted);
  EXPECT_TRUE(c->submit(child.workload, "w", child).value().accepted);
  EXPECT_TRUE(c->submit(grandchild.workload, "w", grandchild).value().accepted);
  EXPECT_EQ(wait_state(*c, upstream.id, {TaskState::Failed, TaskState::Succeeded}, 10000),
            TaskState::Failed);

  // Neither dependent can ever run. Both end instead of staying Blocked, and
  // neither is dispatched.
  EXPECT_EQ(wait_state(*c, child.id, {TaskState::Failed, TaskState::Succeeded}, 3000),
            TaskState::Failed);
  EXPECT_EQ(wait_state(*c, grandchild.id, {TaskState::Failed, TaskState::Succeeded}, 3000),
            TaskState::Failed);
  EXPECT_TRUE(c->inspect(child.id).value().attempts.empty());
  EXPECT_TRUE(c->inspect(grandchild.id).value().attempts.empty());
}

TEST(coordinator_dependents_of_a_cancelled_task_fail) {
  ScriptedFabric f;  // no nodes: nothing is ever dispatched
  f.start();
  auto c = f.client();

  auto upstream = mk_task("cancelled-upstream");
  auto child = mk_task("child");
  child.dependencies = {upstream.id};
  EXPECT_TRUE(c->submit(upstream.workload, "w", upstream).value().accepted);
  EXPECT_TRUE(c->submit(child.workload, "w", child).value().accepted);

  auto rc = c->cancel(upstream.id);
  EXPECT_TRUE(rc.ok());
  EXPECT_TRUE(rc.value().cancelled);
  EXPECT_EQ(wait_state(*c, child.id, {TaskState::Failed, TaskState::Succeeded}, 3000),
            TaskState::Failed);
}

TEST(coordinator_task_submitted_after_its_dependency_succeeded_runs) {
  Fixture f;
  f.start();
  f.add_node(2);
  auto c = f.client();
  EXPECT_TRUE(c->wait_for_nodes(1, 10000));

  auto first = mk_task("first");
  EXPECT_TRUE(c->submit(first.workload, "w", first).value().accepted);
  EXPECT_EQ(wait_state(*c, first.id, {TaskState::Succeeded, TaskState::Failed}, 15000),
            TaskState::Succeeded);

  // The dependency finished before this task existed, so no later success
  // report will ever release it.
  auto second = mk_task("second");
  second.dependencies = {first.id};
  EXPECT_TRUE(c->submit(second.workload, "w", second).value().accepted);
  EXPECT_EQ(wait_state(*c, second.id, {TaskState::Succeeded, TaskState::Failed}, 5000),
            TaskState::Succeeded);
}
