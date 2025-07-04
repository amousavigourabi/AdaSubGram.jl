@testset "Subword splitting" begin
  @test AdaSubGram.Dataset.split_subwords("never", 3, 6) == ["<ne", "nev", "eve", "ver", "er>", "<nev", "neve", "ever", "ver>", "<neve", "never", "ever>", "<never", "never>"]
  @test AdaSubGram.Dataset.split_subwords("éagle", 5, 6) == ["<éagl", "éagle", "agle>", "<éagle", "éagle>"]
end

@testset "Assign labels blank" begin
  @test AdaSubGram.Dataset.assign_labels!(["hello", "my", "name", "is", "jeff"], Dict{String, UInt64}(), Vector{String}(undef, 100), Vector{UInt64}(undef, 100), UInt64(1)) == ([1, 2, 3, 4, 5], 6)
  @test AdaSubGram.Dataset.assign_labels!(["that", "is", "that"], Dict{String, UInt64}(), Vector{String}(undef, 100), Vector{UInt64}(undef, 100), UInt64(1)) == ([1, 2, 1], 3)
end

@testset "Assign labels with existing" begin
  @test AdaSubGram.Dataset.assign_labels!(["hello", "my", "name", "is", "jeff"], Dict{String, UInt64}("hello" => 3, "jeff" => 1, "is" => 2), ["jeff", "is", "hello", "", ""], UInt64[1, 1, 1, 0, 0], UInt64(4)) == ([3, 4, 5, 2, 1], 6)
  @test AdaSubGram.Dataset.assign_labels!(["that", "is", "that"], Dict{String, UInt64}("this" => 1, "is" => 2), ["this", "is", ""], UInt64[0, 1, 0], UInt64(3)) == ([3, 2, 3], 4)
end

@testset "Assign labels updates maps" begin
  index_to_string = ["jeff", "is", "hello", "", ""]
  counts = UInt64[1, 2, 1, 0, 0, 0]
  string_to_index = Dict{String, UInt64}("hello" => 3, "jeff" => 1, "is" => 2)
  AdaSubGram.Dataset.assign_labels!(["hello", "my", "name", "is", "jeff"], string_to_index, index_to_string, counts, UInt64(4))
  @test index_to_string == ["jeff", "is", "hello", "my", "name"]
  @test counts == [2, 3, 2, 1, 1, 0]
  @test string_to_index == Dict{String, UInt64}("hello" => 3, "jeff" => 1, "is" => 2, "my" => 4, "name" => 5)
end

@testset "Pair contexts" begin
  @test AdaSubGram.Dataset.pair_contexts(UInt64[0, 1, 2, 3], 1) == [(0, [1]), (1, [0, 2]), (2, [1, 3]), (3, [2])]
  @test AdaSubGram.Dataset.pair_contexts(UInt64[0, 1], 1) == [(0, [1]), (1, [0])]
  @test AdaSubGram.Dataset.pair_contexts(UInt64[0, 1], 0) == [(0, []), (1, [])]
  @test AdaSubGram.Dataset.pair_contexts(UInt64[0], 1) == [(0, [])]
  @test AdaSubGram.Dataset.pair_contexts(UInt64[], 1) == []
  @test AdaSubGram.Dataset.pair_contexts(UInt64[0, 1, 0, 2, 3, 5], 2) == [(0, [1, 0]), (1, [0, 0, 2]), (0, [0, 1, 2, 3]), (2, [1, 0, 3, 5]), (3, [0, 2, 5]), (5, [2, 3])]
end
