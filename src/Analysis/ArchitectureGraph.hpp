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
// This file defines a reusable architecture-routing analysis abstraction used
// by path expansion. The graph models legal resource-to-resource routing hops
// and stores hop-local transfer semantics such as transfer units and FIFO
// kinds.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_SCHEDULER_ANALYSIS_ARCHITECTUREGRAPH_HPP
#define KTIR_SCHEDULER_ANALYSIS_ARCHITECTUREGRAPH_HPP

#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace scheduler {

class ArchitectureGraph {
 public:
  using NodeId = unsigned;

  enum class ResourceKind {
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
    StorageTransfer,    // memory-to-memory
    StreamingTransfer,  // memory to compute (and vice versa), or
                        // compute-to-compute
  };

  struct ResourceNode {
    NodeId id;
    Resource resource;
  };

  struct EdgeInfo {
    NodeId source;
    NodeId target;
    EdgeKind edge_kind;
    std::optional<mlir::ktdf::ComponentType>
        transfer_unit;  // load-store unit if applicable
    std::optional<mlir::ktdf::FifoType> fifo_type;
    unsigned cost = 1;
  };

  using Path = llvm::SmallVector<NodeId>;
  using NeighborList = llvm::SmallVector<NodeId>;
  using EdgeList = llvm::SmallVector<EdgeInfo>;

  ArchitectureGraph() = default;

  static ArchitectureGraph createBuiltInSpyre();

  NodeId addNode(Resource resource);
  void addEdge(const EdgeInfo& edge_info);

  bool hasNode(NodeId node_id) const;
  std::optional<ResourceNode> getNode(NodeId node_id) const;

  NeighborList getNeighbors(NodeId node_id) const;
  std::optional<EdgeInfo> getEdgeInfo(NodeId source, NodeId target) const;
  std::optional<Path> findShortestPath(NodeId source, NodeId target) const;

  /// Get the node ID for a given resource.
  NodeId getNodeIdForResource(Resource resource) const;

  void dump(llvm::raw_ostream& os) const;

  static llvm::StringRef stringifyResource(Resource resource);
  static llvm::StringRef stringifyResourceKind(ResourceKind kind);
  static llvm::StringRef stringifyEdgeKind(EdgeKind edge_kind);
  static ResourceKind getResourceKind(Resource resource);

  /// Determine FIFO type from edge info and resources
  /// If edge has explicit fifo_type, uses that; otherwise infers from resources
  static mlir::ktdf::FifoSlotType getFifoTypeFromEdge(
      mlir::MLIRContext* ctx, const EdgeInfo& edge, Resource source_resource,
      Resource dest_resource, mlir::Type element_type,
      llvm::ArrayRef<int64_t> shape);

 private:
  NodeId next_node_id_ = 0;
  llvm::MapVector<NodeId, ResourceNode> nodes_;
  llvm::DenseMap<NodeId, EdgeList> adjacency_;
};

}  // namespace scheduler

#endif  // KTIR_SCHEDULER_ANALYSIS_ARCHITECTUREGRAPH_HPP

// Made with Bob
