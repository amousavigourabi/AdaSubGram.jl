module Import

using AdaSubGram

const Filepath=String

function read_encodings(file::Filepath)::Dict{String, Vector{Float32}}
  dictionary = Dict{String, Vector{Float32}}()
  first_line = true
  for line in eachline(file)
    if first_line
      first_line = false
      continue
    end
    parts = split(line, " ")
    label = parts[1]
    vector = parse.(Float32, parts[2:end])
    dictionary[label] = vector
  end
  return dictionary
end

export read_encodings

end
