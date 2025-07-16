using AdaSubGram
using Test

@testset "All tests" begin
  include("./test_hashing.jl")
  include("./test_preprocessing.jl")
  include("./test_dataset.jl")
  include("./test_model.jl")
  include("./test_hierarchical_softmax.jl")
end
