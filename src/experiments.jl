module Experiments

using AdaSubGram
using LinearAlgebra
using DataStructures

const DATA="C:\\Users\\atour\\Documents\\Projects\\AdaSubGram\\data"

function generate_embeddings(handles::Vector{String})::Nothing
  for handle in handles
    AdaSubGram.create_encodings("$(DATA)\\in\\$(handle).txt", "$(DATA)\\out\\$(handle).vec")
  end
end

function get_closest(handle::String, form::String)::Nothing
  encodings = AdaSubGram.Import.read_encodings("$(DATA)\\out\\$(handle).vec")
  compare_vector = encodings[form]
  maxheap = MutableBinaryMaxHeap{Tuple{Float32, String}}()
  for (label, vector) in encodings
    if (label == form)
      continue
    end
    cosine_similarity = dot(vector, compare_vector)
    push!(maxheap, (cosine_similarity, label))
  end
  for i in 1:10
    println("$(i)th is $(pop!(maxheap)[2]).")
  end
end

export generate_embeddings, get_closest

end
