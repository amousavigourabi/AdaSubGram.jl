module Import

using AdaSubGram

const Filepath=String

function read_encodings(file::Filepath)::Dict{String, Vector{Float32}}
  dictionary = Dict{String, Vector{Float32}}()
  for line in eachline(file)
    parts = split(line, " "; limit=2)
    label = parts[1]
    vector = parse.(Float32, split(parts[2]))
    dictionary[label] = vector
  end
  return dictionary
end

export read_encodings

end
