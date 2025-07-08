module Export

using AdaSubGram

const Filepath=String

function get_vectors(word::UInt64, subwords::Vector{UInt32}, subword_weights::AbstractArray{Float32, 2}, sense_weights::AbstractArray{Float32, 3})::Array{Float32, 2}
  @inbounds return sense_weights[word, :, :]' .+ sum(subword_weights[subwords, :], dims=1)
end

function format_vector(vector::Vector{Float32}, word::String, i::Int64)::String
  return "$(word)_$i $(join(vector, " "))\n"
end

function embeddings(output::Filepath, labels::Dict{String, UInt64}, subword_weights::AbstractArray{Float32, 2}, sense_weights::AbstractArray{Float32, 3}, s_min::Int64, s_max::Int64, n::UInt32)
  open(output, "w") do file
    @inbounds write(file, "$(length(labels)*size(sense_weights)[3]) $(size(subword_weights)[2])\n")
    for (word, id) in labels
      vectors = get_vectors(id, AdaSubGram.Hashing.hash_words(AdaSubGram.Dataset.split_subwords(word, s_min, s_max), n), subword_weights, sense_weights)
      @inbounds for (i, j) in enumerate(1:size(vectors)[1])
        @inbounds write(file, format_vector(vectors[j, :], word, i))
      end
    end
  end
end

export embeddings

end
