module Export

using AdaSubGram

const Filepath=String

function get_vectors(word::UInt64, subwords::Vector{UInt32}, subword_weights::AbstractArray{Float32, 2}, sense_weights::AbstractArray{Float32, 3})::Array{Float32, 2}
  @inbounds return sense_weights[:, :, word] .+ sum(subword_weights[:, subwords], dims=2)
end

function format_vector(vector::Vector{Float32}, word::String, i::Int64)::String
  return "$(word)_$i $(join(vector, " "))\n"
end

function embeddings(output::Filepath, labels::Dict{String, UInt64}, subword_weights::AbstractArray{Float32, 2}, sense_weights::AbstractArray{Float32, 3}, s_min::Int64, s_max::Int64, n::UInt32)
  failed_cutoff = 0
  open(output, "w") do file
    for (word, id) in labels
      vectors = get_vectors(id, AdaSubGram.Hashing.hash_words(AdaSubGram.Dataset.split_subwords(word, s_min, s_max), n), subword_weights, sense_weights)
      exists_check = Set{Vector{Float32}}()
      @inbounds for i in 1:size(vectors)[2]
        @inbounds if (vectors[:, i] ∈ exists_check || all(map(iszero, vectors[:, i])))
          failed_cutoff += 1
          continue
        end
        @inbounds push!(exists_check, vectors[:, i])
        @inbounds write(file, format_vector(vectors[:, i], word, i))
      end
    end
  end
  println("$(failed_cutoff) vectors were filtered out for not meeting the cutoff.")
end

export embeddings

end
