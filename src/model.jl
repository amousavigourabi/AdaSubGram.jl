module Model

mutable struct Parameters
  theta_subwords::Array{Float32, 2}
  theta_senses::Array{Float32, 3}
  betas::Array{Float32, 2}
end

end
