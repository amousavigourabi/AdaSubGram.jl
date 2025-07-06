module HuffmanTree

using DataStructures

abstract type HuffmanNode end

struct HuffmanLeaf <: HuffmanNode
  freq::Float64
  symbol::UInt64
end

struct HuffmanInternal <: HuffmanNode
  freq::Float64
  left::HuffmanNode
  right::HuffmanNode
end

Base.isless(a::Tuple{UInt64, HuffmanNode}, b::Tuple{UInt64, HuffmanNode}) = isless(a[1], b[1])
Base.:(==)(a::Tuple{UInt64, HuffmanNode}, b::Tuple{UInt64, HuffmanNode}) = a[1] == b[1]

function construct_tree(freqs::Vector{UInt64})::HuffmanNode
  pq = BinaryMinHeap{Tuple{UInt64, HuffmanNode}}()
  for (symbol, freq) in enumerate(freqs)
    push!(pq, (freq, HuffmanLeaf(freq, symbol)))
  end
  while length(pq) > 1
    (f1, n1) = pop!(pq)
    (f2, n2) = pop!(pq)
    new_node = HuffmanInternal(f1 + f2, n1, n2)
    push!(pq, (f1 + f2, new_node))
  end
  return pop!(pq)[2]
end

function extract_codes!(node::HuffmanNode, codes::Vector{Vector{Int32}}, decisions::Vector{Vector{Float32}}, node_ids::Dict{HuffmanNode, Int64}, prefix::Vector{Int32}, decision_prefix::Vector{Float32})::Nothing
  if node isa HuffmanLeaf
    codes[node.symbol] = deepcopy(prefix)
    decisions[node.symbol] = deepcopy(decision_prefix)
  elseif node isa HuffmanInternal
    push!(prefix, node_ids[node])
    push!(decision_prefix, 1.0f0)
    extract_codes!(node.left, codes, decisions, node_ids, prefix, decision_prefix)
    pop!(decision_prefix)
    push!(decision_prefix, 0.0f0)
    extract_codes!(node.right, codes, decisions, node_ids, prefix, decision_prefix)
    pop!(decision_prefix)
    pop!(prefix)
  end
  return nothing
end

function assign_node_ids(node::HuffmanNode, id_counter::Base.RefValue{Int64}, node_ids::Dict{HuffmanNode, Int64})
  if node isa HuffmanInternal
    node_ids[node] = id_counter[]
    id_counter[] += 1
    assign_node_ids(node.left, id_counter, node_ids)
    assign_node_ids(node.right, id_counter, node_ids)
  end
end

function assign_leaf_ids(node::HuffmanNode, id_counter::Base.RefValue{Int64}, node_ids::Dict{HuffmanNode, Int64})
  if node isa HuffmanInternal
    assign_leaf_ids(node.left, id_counter, node_ids)
    assign_leaf_ids(node.right, id_counter, node_ids)
  elseif node isa HuffmanLeaf
    node_ids[node] = id_counter[]
    id_counter[] += 1
  end
end

function build_node_ids(tree::HuffmanNode)::Dict{HuffmanNode, Int64}
  node_ids = Dict{HuffmanNode, Int64}()
  id_counter = Ref(1)
  assign_node_ids(tree, id_counter, node_ids)
  assign_leaf_ids(tree, id_counter, node_ids)
  return node_ids
end

function huffman_paths(counts::Vector{UInt64})::Tuple{Vector{Vector{Int32}}, Vector{Vector{Float32}}}
  tree = construct_tree(counts)
  node_ids = build_node_ids(tree)
  nodes = Vector{Vector{Int32}}(undef, length(counts))
  decisions = Vector{Vector{Float32}}(undef, length(counts))
  extract_codes!(tree, nodes, decisions, node_ids, Int32[], Float32[])
  return nodes, decisions
end

function hierarchical_softmax_loss(results::Vector{Float32}, targets::Vector{Float32})::Float64
  ϵ = 1e-7
  ℓ = 0.0
  for (p, d) in zip(results, targets)
    ℓ -= d * log(p + ϵ) + (1 - d) * log(1 - p + ϵ)
  end
  return ℓ
end

export huffman_paths, hierarchical_softmax_loss

end
