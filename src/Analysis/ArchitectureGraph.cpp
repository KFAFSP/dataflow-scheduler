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
#include "src/Dialects/KTDF/KTDFEnums.cpp.inc"

namespace {
bool isHardwareSupportedGraphFifoType(mlir::ktdf::FifoType fifo_type) {
  switch (fifo_type) {
    case mlir::ktdf::FifoType::SfpToLxsu:
    case mlir::ktdf::FifoType::LxluToSfp:
      return true;
    case mlir::ktdf::FifoType::HbmToSfp:
    case mlir::ktdf::FifoType::SfpToHbm:
      return false;
  }
  llvm_unreachable("unimplemented fifo type");
}
}  // namespace

namespace scheduler {

ArchitectureGraph ArchitectureGraph::createBuiltInSpyre() {
  ArchitectureGraph graph;

  graph.addNode({Resource::HBM, ResourceNodeKind::Memory});
  graph.addNode({Resource::LX, ResourceNodeKind::Memory});
  graph.addNode({Resource::SFP, ResourceNodeKind::Compute});
  graph.addNode({Resource::PE, ResourceNodeKind::Compute});

  graph.addEdge({Resource::HBM, Resource::LX, EdgeKind::StorageTransfer,
                 mlir::ktdf::ComponentType::L3LU, std::nullopt, 1});
  graph.addEdge({Resource::LX, Resource::HBM, EdgeKind::StorageTransfer,
                 mlir::ktdf::ComponentType::L3SU, std::nullopt, 1});
  graph.addEdge({Resource::LX, Resource::SFP, EdgeKind::StreamingTransfer,
                 mlir::ktdf::ComponentType::LXLU,
                 mlir::ktdf::FifoType::LxluToSfp, 1});
  graph.addEdge({Resource::SFP, Resource::LX, EdgeKind::StreamingTransfer,
                 mlir::ktdf::ComponentType::LXSU,
                 mlir::ktdf::FifoType::SfpToLxsu, 1});

  // Longer alternate route retained for future shortest-path testing.
  graph.addEdge({Resource::SFP, Resource::PE, EdgeKind::StreamingTransfer,
                 std::nullopt,
                 std::nullopt,  // todo: add fifo type
                 1});
  graph.addEdge({Resource::PE, Resource::LX, EdgeKind::StreamingTransfer,
                 std::nullopt,
                 std::nullopt,  // todo: add fifo type
                 1});

  return graph;
}

void ArchitectureGraph::addNode(const ResourceNode& node) {
  if (hasNode(node.resource)) return;
  nodes_.push_back(node);
  node_kinds_[node.resource] = node.node_kind;
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

bool ArchitectureGraph::hasNode(Resource resource) const {
  return node_kinds_.contains(resource);
}

std::optional<ArchitectureGraph::ResourceNode> ArchitectureGraph::getNode(
    Resource resource) const {
  auto it = std::find_if(nodes_.begin(), nodes_.end(),
                         [resource](const ResourceNode& node) {
                           return node.resource == resource;
                         });
  if (it == nodes_.end()) return std::nullopt;
  return *it;
}

ArchitectureGraph::NeighborList ArchitectureGraph::getNeighbors(
    Resource resource) const {
  NeighborList neighbors;
  auto it = adjacency_.find(resource);
  if (it == adjacency_.end()) return neighbors;

  for (const EdgeInfo& edge : it->second) {
    neighbors.push_back(edge.target);
  }
  return neighbors;
}

std::optional<ArchitectureGraph::EdgeInfo> ArchitectureGraph::getEdgeInfo(
    Resource source, Resource target) const {
  auto it = adjacency_.find(source);
  if (it == adjacency_.end()) return std::nullopt;

  for (const EdgeInfo& edge : it->second) {
    if (edge.target == target) return edge;
  }
  return std::nullopt;
}

std::optional<ArchitectureGraph::Path> ArchitectureGraph::findShortestPath(
    Resource source, Resource target) const {
  if (!hasNode(source) || !hasNode(target)) return std::nullopt;

  std::queue<Resource> worklist;
  llvm::DenseMap<Resource, Resource> predecessor;
  llvm::DenseMap<Resource, bool> visited;

  worklist.push(source);
  visited[source] = true;

  while (!worklist.empty()) {
    Resource current = worklist.front();
    worklist.pop();

    if (current == target) break;

    auto neighbors = getNeighbors(current);
    for (Resource neighbor : neighbors) {
      if (visited.lookup(neighbor)) continue;
      visited[neighbor] = true;
      predecessor[neighbor] = current;
      worklist.push(neighbor);
    }
  }

  if (!visited.lookup(target)) return std::nullopt;

  Path reversed_path;
  Resource current = target;
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

void ArchitectureGraph::dump(llvm::raw_ostream& os) const {
  os << "ArchitectureGraph {\n";
  os << "  Nodes:\n";
  for (const ResourceNode& node : nodes_) {
    os << "    - " << stringifyResource(node.resource) << " ["
       << stringifyNodeKind(node.node_kind) << "]\n";
  }

  os << "  Edges:\n";
  for (const ResourceNode& node : nodes_) {
    auto it = adjacency_.find(node.resource);
    if (it == adjacency_.end()) continue;
    for (const EdgeInfo& edge : it->second) {
      os << "    - " << stringifyResource(edge.source) << " -> "
         << stringifyResource(edge.target) << " ["
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

llvm::StringRef ArchitectureGraph::stringifyNodeKind(
    ResourceNodeKind node_kind) {
  switch (node_kind) {
    case ResourceNodeKind::Memory:
      return "Memory";
    case ResourceNodeKind::Compute:
      return "Compute";
  }
  llvm_unreachable("unknown ArchitectureGraph::ResourceNodeKind");
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

}  // namespace scheduler

// Made with Bob
