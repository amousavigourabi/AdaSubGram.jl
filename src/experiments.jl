module Experiments

using AdaSubGram

function generate_embeddings(handles::Vector{String})::Nothing
  for handle in handles
    AdaSubGram.create_encodings("C:\\Users\\atour\\Documents\\Projects\\AdaSubGram\\data\\in\\$(handle).txt", "C:\\Users\\atour\\Documents\\Projects\\AdaSubGram\\data\\out\\$(handle).vec")
  end
end

export generate_embeddings

end
