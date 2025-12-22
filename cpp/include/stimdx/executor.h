#pragma once

#include <vector>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "stimdx.pb.h"
#include "stim.h"

namespace stimdx {

/**
 * Runtime state for dynamic circuit execution.
 * Mirrors Python's ExecContext.
 */
struct ExecContext {
    stim::TableauSimulator<stim::MAX_BITWORD_WIDTH> sim;
    std::vector<bool> meas_record;
    std::vector<bool> last_block_meas;
    
    explicit ExecContext(std::optional<uint64_t> seed = std::nullopt)
        : sim(std::mt19937_64(seed.value_or(std::random_device{}()))) {}
};

/**
 * Execute the circuit AST against the given context.
 * @param circuit The protobuf Circuit message
 * @param ctx The execution context (modified in place)
 */
void execute(const Circuit& circuit, ExecContext& ctx);

/**
 * Evaluate a condition against the current context.
 * @param cond The protobuf Condition message
 * @param ctx The current execution context
 * @return true if condition is satisfied
 */
bool eval_condition(const Condition& cond, const ExecContext& ctx);

/**
 * Sample the circuit for multiple shots.
 * @param circuit The protobuf Circuit message
 * @param shots Number of shots to sample
 * @param seed Optional seed for deterministic sampling
 * @return Vector of measurement records (one per shot)
 */
std::vector<std::vector<bool>> sample(
    const Circuit& circuit,
    int shots,
    std::optional<int64_t> seed = std::nullopt
);

/**
 * Sample from a serialized protobuf Circuit.
 * This is the main entry point from Python.
 * @param proto_bytes Serialized Circuit protobuf
 * @param shots Number of shots to sample  
 * @param seed Optional seed for deterministic sampling
 * @return Vector of measurement records
 */
std::vector<std::vector<bool>> sample_from_proto(
    const std::string& proto_bytes,
    int shots,
    std::optional<int64_t> seed = std::nullopt
);

} // namespace stimdx
