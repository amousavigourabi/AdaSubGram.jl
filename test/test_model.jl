@testset "Model initialization" begin
  ϵ = 1e-7
  uint_range = UInt64.(1:100)
  for _ in 1:20
    dims = rand(3:100)
    senses = rand(2:10)
    word_counts = rand(uint_range, rand(30:100))
    subword_counts = rand([1_000, 2_000, 5_000, 10_000, 20_000])
    model = AdaSubGram.Model.initialize(dims, word_counts, subword_counts, senses)
    @test all(-0.5 / dims - ϵ .<= model.in_subwords .<= 0.5 / dims + ϵ)
    @test all(-0.5 / dims - ϵ .<= model.in_senses .<= 0.5 / dims + ϵ)
    @test all(-0.5 / dims - ϵ .<= model.out .<= 0.5 / dims + ϵ)
    @test all(model.ns[1, :] .== word_counts)
    @test all(model.ns[2:senses, :] .== 0.0)
  end
end

# @testset "Sense likelihoods" begin
#   ϵ = 1e-7
#   for _ in 1:100
#     num_senses = rand(3:20)
#     output_dims = rand(50:200)
#     output = rand(Float32, output_dims, num_senses)
#     sense_likelihoods = zeros(Float32, num_senses)
#     AdaSubGram.Model.sense_likelihoods!(sense_likelihoods, 0.5f0)
#     @test 1 - ϵ < sum(sense_likelihoods) < 1 + ϵ
#   end
# end

# @testset "HAHA" begin
#   AdaSubGram.create_encodings("C:\\Users\\atour\\Documents\\word-embeddings\\dataset.txt", "C:\\Users\\atour\\Downloads\\results.vec")
# end
