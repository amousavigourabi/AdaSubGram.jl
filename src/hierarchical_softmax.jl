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
  symbol::Int32
end

Base.isless(a::Tuple{UInt64, HuffmanNode}, b::Tuple{UInt64, HuffmanNode}) = isless(a[1], b[1])
Base.:(==)(a::Tuple{UInt64, HuffmanNode}, b::Tuple{UInt64, HuffmanNode}) = a[1] == b[1]

function construct_tree(freqs::Vector{UInt64})::HuffmanNode
  pq = BinaryMinHeap{Tuple{UInt64, HuffmanNode}}()
  for (symbol, freq) in enumerate(freqs)
    push!(pq, (freq, HuffmanLeaf(freq, symbol)))
  end
  i = Int32(1)
  while length(pq) > 1
    (f1, n1) = pop!(pq)
    (f2, n2) = pop!(pq)
    new_node = HuffmanInternal(f1 + f2, n1, n2, i)
    push!(pq, (f1 + f2, new_node))
    i += 1
  end
  return pop!(pq)[2]
end

function extract_codes!(root::HuffmanNode, codes::Vector{Tuple{Vector{Int32}, Vector{Float32}}})::Nothing
  node_queue = Deque{Tuple{HuffmanNode, Vector{Int32}, Vector{Float32}}}()
  push!(node_queue, (root, Int32[], Float32[]))
  while !isempty(node_queue)
    node, prefix, decision_prefix = popfirst!(node_queue)
    if node isa HuffmanLeaf
      @inbounds codes[node.symbol] = prefix, decision_prefix
    elseif node isa HuffmanInternal
      prefix = deepcopy(prefix)
      decision_prefix_one = deepcopy(decision_prefix)
      push!(prefix, node.symbol)
      push!(decision_prefix_one, 1.0f0)
      push!(node_queue, (node.left, prefix, decision_prefix_one))
      push!(decision_prefix, 0.0f0)
      push!(node_queue, (node.right, prefix, decision_prefix))
    end
  end
  return nothing
end

function huffman_paths(counts::Vector{UInt64})::Vector{Tuple{Vector{Int32}, Vector{Float32}}}
  tree = construct_tree(counts)
  nodes_decisions = Vector{Tuple{Vector{Int32}, Vector{Float32}}}(undef, length(counts))
  extract_codes!(tree, nodes_decisions)
  return nodes_decisions
end

function hierarchical_softmax_loss(results::Array{Float32, 2}, targets::Vector{Float32})::Float64
  ϵ = 1e-7
  ℓ = 0.0
  _, amount_of_results = size(results)
  for i in 1:amount_of_results
    @simd for j in 1:length(targets)
      @inbounds ℓ -= targets[j] * log(results[j, i] + ϵ) + (1 - targets[j]) * log(1 - results[j, i] + ϵ)
    end
  end
  return ℓ
end

export huffman_paths, hierarchical_softmax_loss

end
