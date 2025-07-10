module Model

using AdaSubGram
using LinearAlgebra
using SpecialFunctions
using ProgressMeter
using Base.Threads

"""
    Parameters(in_subwords::Array{Float32, 2}, in_senses::Array{Float32, 3}, out::Array{Float32, 2}, as::Array{Float32, 2}, bs::Array{Float32, 2})

Keeps track of and maintains the model
parameters that we are optimizing.
"""
struct Parameters
  in_subwords::Array{Float32, 2}
  in_senses::Array{Float32, 3}
  out::Array{Float32, 2}
  ns::Array{Float32, 2}
  word_counts::Vector{UInt64}
end

function initialize(vector_dims::Int64, word_counts::Vector{UInt64}, input_subwords::Int64, senses::Int64)
  input_words = length(word_counts)
  in_subwords = (rand(Float32, vector_dims, input_subwords) .- 0.5) ./ vector_dims
  in_senses = (rand(Float32, vector_dims, senses, input_words) .- 0.5) ./ vector_dims
  out = (rand(Float32, vector_dims, input_words - 1) .- 0.5) ./ vector_dims
  ns = zeros(Float32, senses, input_words)
  @views @inbounds ns[1, :] .+= word_counts
  model = Parameters(in_subwords, in_senses, out, ns, word_counts)
  return model
end

function forward_pass!(model::Parameters, input_word::UInt64, input_subwords::Vector{UInt32}, latent::AbstractArray{Float32, 2}, output::AbstractArray{Float32, 2})::Nothing
  @views @inbounds latent .= model.in_senses[:, :, input_word] .+ sum(model.in_subwords[:, input_subwords], dims=2) # allocations
  mul!(output, model.out', latent)
  return nothing
end

# TODO
# function sense_likelihoods(model::Parameters, α::Float32)::Array{Float32, 2}
#   senses, words = size(model.ns)
#   @views as, bs = compute_beta_parameters(model.ns[:, word], α) # TODO compute_beta_parameters new ns config
#   likelihoods = zeros(Float32, words, senses)
#   @simd for sense in 1:senses
#     @views @inbounds likelihoods[:, sense] .= as[:, sense] ./ (as[:, sense] .+ bs[:, sense])
#   end
#   prod = ones(Float32, words)
#   for sense in 1:(senses-1)
#     @views @inbounds prod .*= bs[:, sense] ./ (as[:, sense] .+ bs[:, sense])
#     @views @inbounds likelihoods[:, sense+1] .*= prod
#   end
#   return likelihoods
# end

function compute_beta_parameters(ns::AbstractArray{Float32, 1}, α::Float32)
  as = ones(Float32, length(ns))
  bs = fill(α, length(ns))
  cumulative_ns = zeros(Float32, length(ns))
  i = length(ns)
  for n in reverse(ns)
    @inbounds cumulative_ns[i] += n
    @inbounds as[i] += n
    @inbounds bs[i] += cumulative_ns[i]
    i -= 1
  end
  return as, bs
end

function σ(x::Float32)
  return 1 / (1 + exp(-x))
end

# TODO split up train
function train(model::Parameters, training_data::Vector{Tuple{UInt64, Vector{UInt32}, Vector{UInt64}}}, paths::Vector{Tuple{Vector{Int32}, Vector{Float32}}}, batch_size::Int64, epochs::Int64, α::Float32, λ::Float32)
  num_dims, num_senses, _ = size(model.in_senses)
  _, num_out = size(model.out)
  bϝs = Vector{Vector{Float32}}(undef, num_senses)
  ∇out = Vector{Array{Float32, 2}}(undef, num_senses)
  ∇h = Vector{Array{Float32, 2}}(undef, num_senses)
  ∇in_senses = Vector{Array{Float32, 3}}(undef, num_senses)
  ∇in_subwords = Vector{Array{Float32, 2}}(undef, num_senses)
  latent = Vector{Array{Float32, 2}}(undef, num_senses)
  output = Vector{Array{Float32, 2}}(undef, num_senses)
  word_set = Vector{Set{UInt64}}(undef, num_senses)
  subword_set = Vector{Set{UInt32}}(undef, num_senses)
  word_vector = Vector{Vector{UInt64}}(undef, num_senses)
  subword_vector = Vector{Vector{UInt32}}(undef, num_senses)
  for i in 1:nthreads()
    bϝs[i] = zeros(Float32, num_senses)
    ∇out[i] = zeros(Float32, size(model.out))
    ∇h[i] = zeros(Float32, num_dims, num_senses)
    ∇in_senses[i] = zeros(Float32, size(model.in_senses))
    ∇in_subwords[i] = zeros(Float32, size(model.in_subwords))
    latent[i] = zeros(Float32, num_dims, num_senses)
    output[i] = zeros(Float32, num_out, num_senses)
    word_set[i] = Set{UInt64}()
    subword_set[i] = Set{UInt32}()
    word_vector[i] = UInt64[]
    subword_vector[i] = UInt32[]
  end
  sense_likelihoods = zeros(Float32, num_senses, batch_size)
  for epoch in 1:epochs
    L = 0
    minibatches = AdaSubGram.Dataset.minibatches(training_data, batch_size)
    progress = Progress(length(minibatches), desc="Batches epoch $epoch/$epochs")
    for (b, minibatch) in enumerate(minibatches)
      next!(progress)
      for i in 1:nthreads()
        empty!(word_set[i])
        empty!(subword_set[i])
        fill!(∇out[i], 0.0f0)
        fill!(∇h[i], 0.0f0)
        @views @inbounds fill!(∇in_senses[i][:, :, word_vector[i]], 0.0f0)
        @views @inbounds fill!(∇in_subwords[i][:, subword_vector[i]], 0.0f0)
      end
      fill!(sense_likelihoods, 0.0f0)
      # TODO multithreading
      @threads for (j, (word, subwords, context)) in minibatch
        @views forward_pass!(model, word, subwords, latent[threadid()], output[threadid()]) # allocations
        @inbounds @views as, bs = compute_beta_parameters(model.ns[:, word], α)
        for sense in 1:num_senses
          @inbounds ϝs = digamma(as[sense] + bs[sense])
          @inbounds bϝs[threadid()][sense] = digamma(bs[sense]) - ϝs
          @inbounds logbeta_k = digamma(as[sense]) - ϝs
          @views @inbounds logbeta_complements = sum(bϝs[threadid()][1:(sense-1)])
          @inbounds sense_likelihoods[sense, j] = logbeta_k + logbeta_complements
        end
        # TODO random context dropping
        @simd for context_word in context
          @inbounds nodes, _ = paths[context_word]
          @views @inbounds sense_likelihoods[:, j] .+= sum(output[threadid()][nodes, :])
          @views @inbounds max_outputs = maximum(output[threadid()][nodes, :], dims=1) # allocations
          @views @inbounds sense_likelihoods[:, j] .-= sum(max_outputs .+ log.(sum(exp.(output[threadid()][nodes, :] .- max_outputs), dims=1))) # allocations
        end
        @views @inbounds max_sense = maximum(sense_likelihoods[:, j])
        @views @inbounds sense_likelihoods[:, j] .-= max_sense + log(sum(exp.(sense_likelihoods[:, j] .- max_sense)))
        @views @inbounds sense_likelihoods[:, j] .= exp.(sense_likelihoods[:, j])
        @simd for context_word in context
          # TODO check gradient updates!
          @inbounds nodes, decisions = paths[context_word]
          @views @inbounds results = σ.(output[threadid()][nodes, :]) # allocations
          @views δs = (results .- decisions)' # allocations
          @views @inbounds mul!(∇out[threadid()][:, nodes], latent[threadid()], δs, 1, 1)
          @views @inbounds ∇h[threadid()] .= dropdims(sum(reshape(δs, :, 1, num_senses) .* reshape(model.out[:, nodes], :, num_dims, 1), dims=1), dims=1) .* sense_likelihoods[:, j]' # allocations
          @views @inbounds ∇in_senses[threadid()][:, :, word] .+= ∇h[threadid()]
          @views @inbounds ∇in_subwords[threadid()][:, subwords] .+= sum(∇h[threadid()], dims=2) # allocations
          L += AdaSubGram.HuffmanTree.hierarchical_softmax_loss(results, decisions)
        end
        push!(word_set[threadid()], word)
        for subword in subwords
          push!(subword_set[threadid()], subword)
        end
      end
      scaling_factor = 1 / length(minibatch)
      η = 0.05 * (1 - (epoch - 1 - (b - 1) / length(minibatches)) / epochs)
      ρ = η * scaling_factor
      for (j, (word, _, _)) in minibatch
        @views @inbounds model.ns[:, word] .= (1-η) .* model.ns[:, word] .+ (η*model.word_counts[word]) .* sense_likelihoods[:, j]
      end
      for i in 1:nthreads()
        word_vector[i] = collect(word_set[i])
        subword_vector[i] = collect(subword_set[i])
        @views model.out .-= ρ .* ∇out[i]
        @views @inbounds model.in_senses[:, :, word_vector[i]] .-= ρ .* ∇in_senses[i][:, :, word_vector[i]]
        @views @inbounds model.in_subwords[:, subword_vector[i]] .-= ρ .* ∇in_subwords[i][:, subword_vector[i]]
      end
    end
    L /= length(training_data)
    finish!(progress)
    println("Total training loss at epoch $epoch/$epochs: $L.")
  end
end

end
