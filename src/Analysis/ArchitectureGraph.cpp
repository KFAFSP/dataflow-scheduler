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

#include "ArchitectureGraph.hpp"

#include <algorithm>
#include <cassert>
#include <queue>

#include "llvm/Support/ErrorHandling.h"

namespace {
bool isHardwareSupportedGraphFifoType(mlir::ktdf::FifoType fifo_type) {
  switch (fifo_type) {
    case mlir::ktdf::FifoType::SfpToLxsu:
    case mlir::ktdf::FifoType::LxluToSfp:
    case mlir::ktdf::FifoType::PeToLxsu:
    case mlir::ktdf::FifoType::LxluToPe:
    case mlir::ktdf::FifoType::PeToSfp:
    case mlir::ktdf::FifoType::SfpToPe:
      return true;
    case mlir::ktdf::FifoType::HbmToSfp:
    case mlir::ktdf::FifoType::SfpToHbm:
    case mlir::ktdf::FifoType::HbmToPe:
    case mlir::ktdf::FifoType::PeToHbm:
      return false;
  }
  llvm_unreachable("unimplemented fifo type");
}
}  // namespace

namespace scheduler {

ArchitectureGraph ArchitectureGraph::createBuiltInSpyre() {
  ArchitectureGraph graph;

  NodeId hbm_node = graph.addNode(Resource::HBM);
  NodeId lx_node = graph.addNode(Resource::LX);
  NodeId sfp_node = graph.addNode(Resource::SFP);
  NodeId pe_node = graph.addNode(Resource::PE);

  graph.addEdge({hbm_node, lx_node, EdgeKind::StorageTransfer,
                 mlir::ktdf::ComponentType::L3LU, std::nullopt, 1});
  graph.addEdge({lx_node, hbm_node, EdgeKind::StorageTransfer,
                 mlir::ktdf::ComponentType::L3SU, std::nullopt, 1});
  graph.addEdge({lx_node, sfp_node, EdgeKind::StreamingTransfer,
                 mlir::ktdf::ComponentType::LXLU,
                 mlir::ktdf::FifoType::LxluToSfp, 1});
  graph.addEdge({sfp_node, lx_node, EdgeKind::StreamingTransfer,
                 mlir::ktdf::ComponentType::LXSU,
                 mlir::ktdf::FifoType::SfpToLxsu, 1});
  graph.addEdge({sfp_node, pe_node, EdgeKind::StreamingTransfer, std::nullopt,
                 mlir::ktdf::FifoType::SfpToPe, 1});
  graph.addEdge({pe_node, sfp_node, EdgeKind::StreamingTransfer, std::nullopt,
                 mlir::ktdf::FifoType::PeToSfp, 1});
  graph.addEdge({pe_node, lx_node, EdgeKind::StreamingTransfer, std::nullopt,
                 mlir::ktdf::FifoType::PeToLxsu, 1});
  graph.addEdge({lx_node, pe_node, EdgeKind::StreamingTransfer, std::nullopt,
                 mlir::ktdf::FifoType::LxluToPe, 1});

  return graph;
}

ArchitectureGraph::NodeId ArchitectureGraph::addNode(Resource resource) {
  NodeId node_id = next_node_id_++;
  nodes_.insert({node_id, {node_id, resource}});
  return node_id;
}

void ArchitectureGraph::addEdge(const EdgeInfo& edge_info) {
  assert(hasNode(edge_info.source) &&
         "source node must exist before adding edge");
  assert(hasNode(edge_info.target) &&
         "target node must exist before adding edge");

  // KTDF may temporarily permit broader FIFO types while constructing or
  // legalizing pipelines, but the architecture graph is stricter: it models
  // only FIFOs that exist in the actual hardware routing fabric.
  if (edge_info.fifo_type &&
      !isHardwareSupportedGraphFifoType(*edge_info.fifo_type)) {
    llvm::report_fatal_error(
        "attempted to add unsupported FIFO type to ArchitectureGraph");
  }

  adjacency_[edge_info.source].push_back(edge_info);
}

bool ArchitectureGraph::hasNode(NodeId node_id) const {
  return nodes_.count(node_id) != 0;
}

std::optional<ArchitectureGraph::ResourceNode> ArchitectureGraph::getNode(
    NodeId node_id) const {
  auto it = nodes_.find(node_id);
  if (it == nodes_.end()) return std::nullopt;
  return it->second;
}

ArchitectureGraph::NeighborList ArchitectureGraph::getNeighbors(
    NodeId node_id) const {
  NeighborList neighbors;
  auto it = adjacency_.find(node_id);
  if (it == adjacency_.end()) return neighbors;

  for (const EdgeInfo& edge : it->second) {
    neighbors.push_back(edge.target);
  }
  return neighbors;
}

std::optional<ArchitectureGraph::EdgeInfo> ArchitectureGraph::getEdgeInfo(
    NodeId source, NodeId target) const {
  auto it = adjacency_.find(source);
  if (it == adjacency_.end()) return std::nullopt;

  for (const EdgeInfo& edge : it->second) {
    if (edge.target == target) return edge;
  }
  return std::nullopt;
}

std::optional<ArchitectureGraph::Path> ArchitectureGraph::findShortestPath(
    NodeId source, NodeId target) const {
  if (!hasNode(source) || !hasNode(target)) return std::nullopt;

  std::queue<NodeId> worklist;
  llvm::DenseMap<NodeId, NodeId> predecessor;
  llvm::DenseMap<NodeId, bool> visited;

  worklist.push(source);
  visited[source] = true;

  while (!worklist.empty()) {
    NodeId current = worklist.front();
    worklist.pop();

    if (current == target) break;

    auto neighbors = getNeighbors(current);
    for (NodeId neighbor : neighbors) {
      if (visited.lookup(neighbor)) continue;
      visited[neighbor] = true;
      predecessor[neighbor] = current;
      worklist.push(neighbor);
    }
  }

  if (!visited.lookup(target)) return std::nullopt;

  Path reversed_path;
  NodeId current = target;
  reversed_path.push_back(current);

  while (current != source) {
    auto pred_it = predecessor.find(current);
    if (pred_it == predecessor.end()) return std::nullopt;
    current = pred_it->second;
    reversed_path.push_back(current);
  }

  Path path;
  path.reserve(reversed_path.size());
  for (auto it = reversed_path.rbegin(); it != reversed_path.rend(); ++it) {
    path.push_back(*it);
  }
  return path;
}

ArchitectureGraph::NodeId ArchitectureGraph::getNodeIdForResource(
    Resource resource) const {
  // Search through nodes to find the one with matching resource
  for (const auto& [node_id, node] : nodes_) {
    if (node.resource == resource) {
      return node_id;
    }
  }
  llvm_unreachable("resource not found in graph");
}

void ArchitectureGraph::dump(llvm::raw_ostream& os) const {
  os << "ArchitectureGraph {\n";
  os << "  Nodes:\n";
  for (const auto& [node_id, node] : nodes_) {
    os << "    - " << node_id << ": " << stringifyResource(node.resource)
       << " [" << stringifyResourceKind(getResourceKind(node.resource))
       << "]\n";
  }

  os << "  Edges:\n";
  for (const auto& [node_id, node] : nodes_) {
    auto it = adjacency_.find(node_id);
    if (it == adjacency_.end()) continue;
    for (const EdgeInfo& edge : it->second) {
      auto source_node = getNode(edge.source);
      auto target_node = getNode(edge.target);
      assert(source_node && target_node &&
             "edge endpoint must reference a node");

      os << "    - " << edge.source << ": "
         << stringifyResource(source_node->resource) << " -> " << edge.target
         << ": " << stringifyResource(target_node->resource) << " ["
         << stringifyEdgeKind(edge.edge_kind);
      if (edge.transfer_unit) {
        os << ", transfer_unit="
           << mlir::ktdf::stringifyComponentType(*edge.transfer_unit);
      }
      if (edge.fifo_type) {
        os << ", fifo_type=" << mlir::ktdf::stringifyFifoType(*edge.fifo_type);
      }
      os << ", cost=" << edge.cost << "]\n";
    }
  }
  os << "}\n";
}

llvm::StringRef ArchitectureGraph::stringifyResource(Resource resource) {
  switch (resource) {
    case Resource::HBM:
      return "HBM";
    case Resource::LX:
      return "LX";
    case Resource::SFP:
      return "SFP";
    case Resource::PE:
      return "PE";
  }
  llvm_unreachable("unknown ArchitectureGraph::Resource");
}

ArchitectureGraph::ResourceKind ArchitectureGraph::getResourceKind(
    Resource resource) {
  switch (resource) {
    case Resource::HBM:
    case Resource::LX:
      return ResourceKind::Memory;
    case Resource::SFP:
    case Resource::PE:
      return ResourceKind::Compute;
  }
  llvm_unreachable("unknown ArchitectureGraph::Resource");
}

llvm::StringRef ArchitectureGraph::stringifyResourceKind(ResourceKind kind) {
  switch (kind) {
    case ResourceKind::Memory:
      return "Memory";
    case ResourceKind::Compute:
      return "Compute";
  }
  llvm_unreachable("unknown ArchitectureGraph::ResourceKind");
}

llvm::StringRef ArchitectureGraph::stringifyEdgeKind(EdgeKind edge_kind) {
  switch (edge_kind) {
    case EdgeKind::StorageTransfer:
      return "StorageTransfer";
    case EdgeKind::StreamingTransfer:
      return "StreamingTransfer";
  }
  llvm_unreachable("unknown ArchitectureGraph::EdgeKind");
}

mlir::ktdf::FifoSlotType ArchitectureGraph::getFifoTypeFromEdge(
    mlir::MLIRContext* ctx, const EdgeInfo& edge, Resource source_resource,
    Resource dest_resource, mlir::Type element_type,
    llvm::ArrayRef<int64_t> shape) {
  // Determine FIFO type from edge
  mlir::ktdf::FifoType fifo_type;
  if (edge.fifo_type) {
    fifo_type = *edge.fifo_type;
  } else {
    // Default based on source/dest resources
    if (source_resource == Resource::HBM && dest_resource == Resource::SFP) {
      fifo_type = mlir::ktdf::FifoType::HbmToSfp;
    } else if (source_resource == Resource::SFP &&
               dest_resource == Resource::HBM) {
      fifo_type = mlir::ktdf::FifoType::SfpToHbm;
    } else if (source_resource == Resource::HBM &&
               dest_resource == Resource::PE) {
      fifo_type = mlir::ktdf::FifoType::HbmToPe;
    } else if (source_resource == Resource::PE &&
               dest_resource == Resource::HBM) {
      fifo_type = mlir::ktdf::FifoType::PeToHbm;
    } else if (source_resource == Resource::LX &&
               dest_resource == Resource::SFP) {
      fifo_type = mlir::ktdf::FifoType::LxluToSfp;
    } else if (source_resource == Resource::SFP &&
               dest_resource == Resource::LX) {
      fifo_type = mlir::ktdf::FifoType::SfpToLxsu;
    } else if (source_resource == Resource::LX &&
               dest_resource == Resource::PE) {
      fifo_type = mlir::ktdf::FifoType::LxluToPe;
    } else if (source_resource == Resource::PE &&
               dest_resource == Resource::LX) {
      fifo_type = mlir::ktdf::FifoType::PeToLxsu;
    } else if (source_resource == Resource::SFP &&
               dest_resource == Resource::PE) {
      fifo_type = mlir::ktdf::FifoType::SfpToPe;
    } else if (source_resource == Resource::PE &&
               dest_resource == Resource::SFP) {
      fifo_type = mlir::ktdf::FifoType::PeToSfp;
    } else {
      llvm_unreachable("Unsupported FIFO edge");
    }
  }

  // Calculate total number of elements
  int64_t num_elements = 1;
  for (int64_t dim : shape) {
    num_elements *= dim;
  }

  return mlir::ktdf::FifoSlotType::get(ctx, fifo_type, num_elements,
                                       element_type);
}

}  // namespace scheduler

// Made with Bob
