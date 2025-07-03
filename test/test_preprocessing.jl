@testset "String normalization" begin
  @test AdaSubGram.Preprocessing.normalize_encodings("d") == "d"
  @test AdaSubGram.Preprocessing.normalize_encodings("é") == "é"
  @test AdaSubGram.Preprocessing.normalize_encodings("é") == "é"
end
