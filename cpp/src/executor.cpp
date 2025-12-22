#include "stimdx/executor.h"
#include <sstream>

namespace stimdx {

void execute(const Circuit &circuit, ExecContext &ctx) {
  for (const auto &node : circuit.nodes()) {
    if (node.has_stim_block()) {
      const auto &block = node.stim_block();

      // Get measurement count before executing
      size_t before_len = ctx.sim.measurement_record.storage.size();

      // Parse and execute the stim circuit
      stim::Circuit stim_circuit(block.stim_circuit_text().c_str());
      ctx.sim.safe_do_circuit(stim_circuit);

      // Extract new measurements
      const auto &full_record = ctx.sim.measurement_record.storage;
      std::vector<bool> new_meas(full_record.begin() + before_len,
                                 full_record.end());

      // Append to measurement record
      ctx.meas_record.insert(ctx.meas_record.end(), new_meas.begin(),
                             new_meas.end());

      // Capture as last block measurements if needed
      if (block.capture_as_last()) {
        ctx.last_block_meas = std::move(new_meas);
      }

    } else if (node.has_if_node()) {
      const auto &if_node = node.if_node();
      if (eval_condition(if_node.condition(), ctx)) {
        execute(if_node.body(), ctx);
      }

    } else if (node.has_while_node()) {
      const auto &while_node = node.while_node();
      int iterations = 0;
      int max_iter = while_node.max_iter() > 0 ? while_node.max_iter() : 10000;

      while (eval_condition(while_node.condition(), ctx)) {
        iterations++;
        if (iterations > max_iter) {
          throw std::runtime_error("While-loop exceeded max_iter=" +
                                   std::to_string(max_iter));
        }
        execute(while_node.body(), ctx);
      }

    } else if (node.has_do_while_node()) {
      const auto &do_while = node.do_while_node();
      int iterations = 0;
      int max_iter = do_while.max_iter() > 0 ? do_while.max_iter() : 10000;

      do {
        iterations++;
        if (iterations > max_iter) {
          throw std::runtime_error("Do-While loop exceeded max_iter=" +
                                   std::to_string(max_iter));
        }
        execute(do_while.body(), ctx);
      } while (eval_condition(do_while.condition(), ctx));

    } else {
      throw std::runtime_error("Unknown node type in circuit");
    }
  }
}

bool eval_condition(const Condition &cond, const ExecContext &ctx) {
  if (cond.has_last_meas()) {
    int index = cond.last_meas().index();
    if (index < 0 || static_cast<size_t>(index) >= ctx.last_block_meas.size()) {
      throw std::out_of_range("LastMeas index " + std::to_string(index) +
                              " out of range for last block of size " +
                              std::to_string(ctx.last_block_meas.size()));
    }
    return ctx.last_block_meas[index];

  } else if (cond.has_meas_parity()) {
    int parity = 0;
    int total_len = static_cast<int>(ctx.meas_record.size());

    for (int i : cond.meas_parity().indices()) {
      // Support negative indexing
      int actual_idx = i < 0 ? total_len + i : i;
      if (actual_idx < 0 || actual_idx >= total_len) {
        throw std::out_of_range("MeasParity index " + std::to_string(i) +
                                " out of range for record of size " +
                                std::to_string(total_len));
      }
      if (ctx.meas_record[actual_idx]) {
        parity ^= 1;
      }
    }
    return parity == 1;

  } else {
    throw std::runtime_error("Unknown condition type");
  }
}

std::vector<std::vector<bool>> sample(const Circuit &circuit, int shots,
                                      std::optional<int64_t> seed) {
  std::vector<std::vector<bool>> all_samples;
  all_samples.reserve(shots);

  for (int s = 0; s < shots; ++s) {
    // Create fresh context for each shot
    std::optional<uint64_t> current_seed;
    if (seed.has_value()) {
      current_seed = static_cast<uint64_t>(seed.value() + s);
    }

    ExecContext ctx(current_seed);
    execute(circuit, ctx);
    all_samples.push_back(std::move(ctx.meas_record));
  }

  return all_samples;
}

std::vector<std::vector<bool>> sample_from_proto(const std::string &proto_bytes,
                                                 int shots,
                                                 std::optional<int64_t> seed) {
  Circuit circuit;
  if (!circuit.ParseFromString(proto_bytes)) {
    throw std::runtime_error("Failed to parse Circuit protobuf");
  }
  return sample(circuit, shots, seed);
}

} // namespace stimdx
