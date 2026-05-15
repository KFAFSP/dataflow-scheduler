//===----------------------------------------------------------------------===//
//
// Copyright 2026 The KTIR Scheduler Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
// Architecture Graph
//
// This file defines a reusable architecture-routing analysis abstraction used by
// path expansion. The graph models legal resource-to-resource routing hops and
// stores hop-local transfer semantics such as transfer units and FIFO kinds.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_SCHEDULER_ANALYSIS_ARCHITECTUREGRAPH_HPP
#define KTIR_SCHEDULER_ANALYSIS_ARCHITECTUREGRAPH_HPP

#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace scheduler {

class ArchitectureGraph {
 public:
  enum class ResourceNodeKind {
    Memory,
    Compute,
  };

  enum class Resource {
    HBM,
    LX,
    SFP,
    PE,
  };

  enum class EdgeKind {
    StorageTransfer, // memory-to-memory
    StreamingTransfer, // memory to compute (and vice versa), or compute-to-compute
  };

  struct ResourceNode {
    Resource resource;
    ResourceNodeKind node_kind;
  };

  struct EdgeInfo {
    Resource source;
    Resource target;
    EdgeKind edge_kind;
    std::optional<mlir::ktdf::ComponentType>
        transfer_unit;  // load-store unit if applicable
    std::optional<mlir::ktdf::FifoType> fifo_type;
    unsigned cost = 1;
  };

  using Path = llvm::SmallVector<Resource>;
  using NeighborList = llvm::SmallVector<Resource>;
  using EdgeList = llvm::SmallVector<EdgeInfo>;

  ArchitectureGraph() = default;

  static ArchitectureGraph createBuiltInSpyre();

  void addNode(const ResourceNode &node);
  void addEdge(const EdgeInfo &edge_info);

  bool hasNode(Resource resource) const;
  std::optional<ResourceNode> getNode(Resource resource) const;

  NeighborList getNeighbors(Resource resource) const;
  std::optional<EdgeInfo> getEdgeInfo(Resource source, Resource target) const;
  std::optional<Path> findShortestPath(Resource source,
                                       Resource target) const;

  void dump(llvm::raw_ostream &os) const;

  static llvm::StringRef stringifyResource(Resource resource);
  static llvm::StringRef stringifyNodeKind(ResourceNodeKind node_kind);
  static llvm::StringRef stringifyEdgeKind(EdgeKind edge_kind);

 private:
  llvm::SmallVector<ResourceNode> nodes_;
  llvm::DenseMap<Resource, ResourceNodeKind> node_kinds_;
  llvm::DenseMap<Resource, EdgeList> adjacency_;
};

}  // namespace scheduler

#endif  // KTIR_SCHEDULER_ANALYSIS_ARCHITECTUREGRAPH_HPP

// Made with Bob
