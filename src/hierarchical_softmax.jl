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

function extract_codes!(node::HuffmanNode, codes::Vector{Vector{Int32}}, node_ids::Dict{HuffmanNode, Int64}, prefix::Vector{Int32})::Nothing
  if node isa HuffmanLeaf
    push!(prefix, node_ids[node])
    codes[node.symbol] = deepcopy(prefix)
    pop!(prefix)
  elseif node isa HuffmanInternal
    push!(prefix, node_ids[node])
    extract_codes!(node.left, codes, node_ids, prefix)
    extract_codes!(node.right, codes, node_ids, prefix)
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

function huffman_paths(counts::Vector{UInt64})::Vector{Vector{Int32}}
  tree = construct_tree(counts)
  node_ids = build_node_ids(tree)
  paths = Vector{Vector{Int32}}(undef, length(counts))
  extract_codes!(tree, paths, node_ids, Int32[])
  return paths
end

export huffman_paths

end
