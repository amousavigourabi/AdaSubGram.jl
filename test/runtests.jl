using AdaSubGram
using Test

@testset "Hashing without clipping" begin
  n = 0xffffffff
  @test AdaSubGram.Hashing.fnv_1a_hash("abcdefg", n) == 0x2a9eb737
  @test AdaSubGram.Hashing.fnv_1a_hash("SHo rt", n) == 0x164acfff
  @test AdaSubGram.Hashing.fnv_1a_hash("long-long,word", n) == 0x43c7a9a8
  @test AdaSubGram.Hashing.fnv_1a_hash("spécialchąɾaçtərʂ", n) == 0xaf72b63d
  @test AdaSubGram.Hashing.fnv_1a_hash("ඞ", n) == 0x8137916f
end

@testset "Hashing with clipping" begin
  @test AdaSubGram.Hashing.fnv_1a_hash("abcdEfg ", UInt32(2_000_000)) == 0x10a795
  @test AdaSubGram.Hashing.fnv_1a_hash("abcdEfg ", UInt32(0xffff)) == 0x6a10
  @test AdaSubGram.Hashing.fnv_1a_hash("abcdEfg ", UInt32(0xff)) == 0x7a
  @test AdaSubGram.Hashing.fnv_1a_hash("abcdEfg ", UInt32(2)) == 0x1
  @test AdaSubGram.Hashing.fnv_1a_hash("anything", UInt32(1)) == 0x0
  @test AdaSubGram.Hashing.fnv_1a_hash("ඞ", UInt32(0xabcd)) == 0x8920
end

@testset "Hashing list" begin
  words = ["abcdefg", "BIG WORD", "_ඞ", "spécιαlchąɾaçtərʂ"]
  n = UInt32(1_000_000)
  @test AdaSubGram.Hashing.hash_words(words, n) == AdaSubGram.Hashing.fnv_1a_hash.(words, n)
end
