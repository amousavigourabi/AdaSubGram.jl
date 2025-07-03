@testset "String normalization" begin
  @test AdaSubGram.Preprocessing.normalize("d") == "d"
  @test AdaSubGram.Preprocessing.normalize("é") == "é"
  @test AdaSubGram.Preprocessing.normalize("é") == "é"
end

@testset "Subword splitting" begin
  @test AdaSubGram.Preprocessing.split_subwords("never", 3, 6) == ["<ne", "nev", "eve", "ver", "er>", "<nev", "neve", "ever", "ver>", "<neve", "never", "ever>", "<never", "never>"]
  @test AdaSubGram.Preprocessing.split_subwords("éagle", 5, 6) == ["<éagl", "éagle", "agle>", "<éagle", "éagle>"]
end
