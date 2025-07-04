@testset "Huffman tree construction" begin
  uint_range = UInt64.(1:100)
  for _ in 1:10
    counts = rand(uint_range, rand(20:50))
    output = AdaSubGram.HuffmanTree.huffman_paths(counts)
    sorted_counts = sort(collect(enumerate(counts)), by = x -> x[2])
    @test length(output[sorted_counts[1][1]]) >= length(output[sorted_counts[end][1]])
  end
end
