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

function hierarchical_softmax_loss(results::O, decisions::Vector{Float32}, likelihoods::A, sense_sums::A)::Float64 where {A <: AbstractArray{Float32, 1}, O <: AbstractArray{Float32, 2}}
  fill!(sense_sums, 0.0f0)
  for w in axes(results, 2)
    @simd ivdep for v in axes(results, 1)
      @inbounds sense_sums[w] += log((1.0f0 - decisions[v]) * (1.0f0 - results[v, w]) + decisions[v] * results[v, w])
    end
  end
  sense_sums .*= likelihoods
  return -sum(sense_sums)
end

export huffman_paths, hierarchical_softmax_loss

end
