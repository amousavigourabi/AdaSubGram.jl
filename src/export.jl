module Export

using AdaSubGram
using LinearAlgebra

const Filepath=String

function get_vectors(word::UInt64, subwords::Vector{UInt32}, subword_weights::AbstractArray{Float32, 2}, sense_weights::AbstractArray{Float32, 3})::Array{Float32, 2}
  new_sense_weights = zeros(Float32, size(sense_weights)[1], size(sense_weights)[2])
  @inbounds subsum = sum(subword_weights[:, subwords], dims=2)
  for sense in 1:size(sense_weights)[2]
    if (!all(map(iszero, sense_weights[:, sense, word])))
      @inbounds new_sense_weights[:, sense] .= sense_weights[:, sense, word] .+ subsum
    end
  end
  return new_sense_weights
end

function format_vector(vector::Vector{Float32}, word::String, i::Int64)::String
  return "$(word)_$i $(join(vector, " "))\n"
end

function embeddings(output::Filepath, labels::Dict{String, UInt64}, subword_weights::AbstractArray{Float32, 2}, sense_weights::AbstractArray{Float32, 3}, s_min::Int64, s_max::Int64, n::UInt32, learned_vectors::Int64)
  written_vectors = 0
  open(output, "w") do file
    write(file, "$(learned_vectors) $(size(sense_weights)[1])\n") # HEADER
    for (word, id) in labels
      vectors = get_vectors(id, AdaSubGram.Hashing.hash_words(AdaSubGram.Dataset.split_subwords(word, s_min, s_max), n), subword_weights, sense_weights)
      exists_check = Set{Vector{Float32}}()
      @inbounds for i in 1:size(vectors)[2]
        @inbounds if (vectors[:, i] ∈ exists_check || all(map(iszero, vectors[:, i])))
          continue
        end
        @inbounds push!(exists_check, vectors[:, i])
        vectors[:, i] ./= norm(vectors[:, i])
        @inbounds write(file, format_vector(vectors[:, i], word, i))
        written_vectors += 1
      end
    end
  end
  if (learned_vectors != written_vectors)
    error("The amount of learned and written vectors are not equal ($(learned_vectors) != $(written_vectors))")
  end
end

export embeddings

end
