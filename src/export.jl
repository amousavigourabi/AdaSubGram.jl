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
  open(output, "w") do file
    dims, senses, _ = size(sense_weights)
    @inbounds write(file, "$(length(labels)*senses) $dims\n")
    for (word, id) in labels
      vectors = get_vectors(id, AdaSubGram.Hashing.hash_words(AdaSubGram.Dataset.split_subwords(word, s_min, s_max), n), subword_weights, sense_weights)
      @inbounds for i in 1:size(vectors)[2]
        @inbounds write(file, format_vector(vectors[:, i], word, i))
      end
    end
  end
end

export embeddings

end
