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
