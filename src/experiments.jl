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
  minheap = MutableBinaryMinHeap{Tuple{Float32, String}, DataStructures.FasterForward}()
  for (label, vector) in encodings
    if (label == form)
      continue
    end
    cosine_similarity = dot(vector, compare_vector)
    push!(minheap, (cosine_similarity, label))
  end
  for i in 1:10
    println("$(i)th is $(popmin!(minheap[2])).")
  end
end

export generate_embeddings, get_closest

end
