@testset "Huffman tree construction" begin
  uint_range = UInt64.(1:100)
  for _ in 1:10
    counts = rand(uint_range, rand(20:50))
    nodes_decisions = AdaSubGram.HuffmanTree.huffman_paths(counts)
    sorted_counts = sort(collect(enumerate(counts)), by = x -> x[2])
    @test length(nodes_decisions[sorted_counts[1][1]][1]) >= length(nodes_decisions[sorted_counts[end][1]][1])
    @test all(x -> all(y -> y in (0, 1), x[2]), nodes_decisions)
  end
end

@testset "Loss one-hot likelihoods" begin
  ϵ = 1e-6
  likelihoods = [1.0f0, 0.0f0, 0.0f0]
  results = [1.0f0 0.8f0 0.9f0; 0.5f0 0.7f0 0.1f0; 0.1f0 0.9f0 0.7f0; 0.3f0 1.0f0 0.2f0]
  targets = [1.0f0, 1.0f0, 1.0f0, 0.0f0]
  loss = AdaSubGram.HuffmanTree.hierarchical_softmax_loss(results, targets, likelihoods)
  loss_target = - log(0.5f0) - log(0.1f0) - log(0.7f0)
  @test loss_target - ϵ < loss < loss_target + ϵ
end

@testset "Loss mixed likelihoods" begin
  ϵ = 1e-6
  likelihoods = [0.5f0, 0.2f0, 0.3f0]
  results = [0.8f0 0.9f0 1.0f0; 0.4f0 0.7f0 0.1f0; 0.2f0 1.0f0 0.7f0; 0.0f0 0.6f0 0.2f0]
  targets = [1.0f0, 1.0f0, 1.0f0, 0.0f0]
  loss = AdaSubGram.HuffmanTree.hierarchical_softmax_loss(results, targets, likelihoods)
  loss_target = - 0.5f0 * (log(0.8f0) + log(0.4f0) + log(0.2f0)) - 0.2f0 * (log(0.9f0) + log(0.7f0) + log(0.4f0)) - 0.3f0 * (log(0.1f0) + log(0.7f0) + log(0.8f0))
  @test loss_target - ϵ < loss < loss_target + ϵ
end
