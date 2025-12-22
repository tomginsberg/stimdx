#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stimdx/executor.h"

namespace py = pybind11;

PYBIND11_MODULE(_stimdx_cpp, m) {
  m.doc() = "C++ execution engine for stimdx dynamic circuits";

  m.def(
      "sample_circuit_proto",
      [](const py::bytes &proto_bytes, int shots, std::optional<int64_t> seed) {
        std::string bytes_str = proto_bytes;
        return stimdx::sample_from_proto(bytes_str, shots, seed);
      },
      py::arg("proto_bytes"), py::arg("shots"), py::arg("seed") = py::none(),
      R"doc(
            Sample from a serialized Circuit protobuf.
            
            Args:
                proto_bytes: Serialized Circuit protobuf bytes
                shots: Number of shots to sample
                seed: Optional seed for deterministic sampling
                
            Returns:
                List of measurement records (one list of bools per shot)
        )doc");

  m.def(
      "get_version", []() { return "0.2.0"; },
      "Get the stimdx C++ module version");
}
