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
    @test all(model.ns[:, 1] .== word_counts)
    @test all(model.ns[:, 2:senses] .== 0.0)
  end
end

const documents=[
  "The happy fox jumped over the river.",
  "STATELY, PLUMP BUCK MULLIGAN CAME FROM THE STAIRHEAD, bearing a bowl of lather on which a mirror and a razor lay crossed. A yellow dressing gown, ungirdled, was sustained gently-behind him by the mild morning air. He held the bowl aloft and intoned:",
  "No way mate, that's absolutely insane!"
]

@testset "SCRATCH TRAIN" begin
  dims = 3
  α = 0.5f0
  λ = 0.05f0
  senses = 6
  context = 5
  subword_truncation = 10_000
  s_min = 4
  s_max = 7
  batch_size = 32
  epochs = 100
  normalized_documents = AdaSubGram.Preprocessing.normalize.(documents)
  tokenized_documents = AdaSubGram.Preprocessing.tokenize.(normalized_documents)
  dataset, counts = AdaSubGram.Dataset.create_dataset(tokenized_documents, context, s_min, s_max, UInt32(subword_truncation), UInt64(1_000))
  nodes, decisions = AdaSubGram.HuffmanTree.huffman_paths(counts)
  model = AdaSubGram.Model.initialize(dims, counts, subword_truncation, senses)
  AdaSubGram.Model.train(model, dataset, collect(zip(nodes, decisions)), batch_size, epochs, α, λ)
end
