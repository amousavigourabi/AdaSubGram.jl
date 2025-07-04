@testset "Model initialization" begin
  ϵ = 1e-7
  for _ in 1:20
    dims = rand(3:100)
    alpha = Float32(rand() / 2 + 0.3)
    word_number = rand(30:100)
    senses = rand(2:10)
    word_counts = rand(1:100, word_number)
    subword_counts = rand([1_000, 2_000, 5_000, 10_000, 20_000])
    tree_nodes = Int64(ceil(log2(word_number)))
    model = AdaSubGram.Model.initialize(dims, word_number, word_counts, subword_counts, senses, tree_nodes, alpha)
    @test all(-0.5 / dims - ϵ .<= model.in_subwords .<= 0.5 / dims + ϵ)
    @test all(-0.5 / dims - ϵ .<= model.in_senses .<= 0.5 / dims + ϵ)
    @test all(-0.5 / dims - ϵ .<= model.out .<= 0.5 / dims + ϵ)
    @test all(model.as[:, 1] - word_counts .== 1.0)
    @test all(model.as[:, 2:senses] .== 1.0)
    @test all(model.bs .== alpha)
  end
end
