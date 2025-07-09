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

function forward_pass!(model::Parameters, input_word::UInt64, input_subwords::Vector{UInt32}, latent::AbstractArray{Float32, 2}, output::AbstractArray{Float32, 2}, model_out::AbstractArray{Float32, 2})::Nothing
  @views @inbounds latent .= model.in_senses[:, :, input_word] .+ sum(model.in_subwords[:, input_subwords], dims=2)
  mul!(output, model_out', latent)
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
  ϝs = zeros(Float32, num_senses)
  bϝs = zeros(Float32, num_senses)
  ∇out = zeros(Float32, size(model.out))
  ∇h = zeros(Float32, num_dims, num_senses)
  ∇in_senses = zeros(Float32, size(model.in_senses))
  ∇in_subwords = zeros(Float32, size(model.in_subwords))
  latent = zeros(Float32, num_dims, num_senses)
  output = zeros(Float32, num_out, num_senses)
  for epoch in 1:epochs
    L = 0
    minibatches = AdaSubGram.Dataset.minibatches(training_data, batch_size)
    progress = Progress(length(minibatches), desc="Batches epoch $epoch/$epochs")
    sense_likelihoods = zeros(Float32, num_senses, batch_size)
    for minibatch in minibatches
      next!(progress)
      fill!(∇out, 0.0f0)
      fill!(∇h, 0.0f0)
      fill!(∇in_senses, 0.0f0)
      fill!(∇in_subwords, 0.0f0)
      fill!(sense_likelihoods, 0.0f0)
      for (j, (word, subwords, context)) in enumerate(minibatch)
        @views forward_pass!(model, word, subwords, latent, output, model.out)
        @inbounds @views as, bs = compute_beta_parameters(model.ns[:, word], α)
        for sense in 1:num_senses
          @inbounds ϝs[sense] = digamma(as[sense] + bs[sense])
          @inbounds bϝs[sense] = digamma(bs[sense]) - ϝs[sense]
          @inbounds logbeta_k = digamma(as[sense]) - ϝs[sense]
          @views @inbounds logbeta_complements = sum(bϝs[1:(sense-1)])
          @inbounds sense_likelihoods[sense, j] = logbeta_k + logbeta_complements
        end
        @simd for context_word in context
          @inbounds nodes, _ = paths[context_word]
          @views @inbounds sense_likelihoods[:, j] .+= sum(output[nodes, :])
          @views @inbounds max_outputs = maximum(output[nodes, :], dims=1)
          @views @inbounds sense_likelihoods[:, j] .-= sum(max_outputs .+ log.(sum(exp.(output[nodes, :] .- max_outputs), dims=1)))
        end
        @views @inbounds max_sense = maximum(sense_likelihoods[:, j])
        @views @inbounds sense_likelihoods[:, j] .-= max_sense + log(sum(exp.(sense_likelihoods[:, j] .- max_sense)))
        @views @inbounds sense_likelihoods[:, j] .= exp.(sense_likelihoods[:, j])
        @simd for context_word in context
          @inbounds nodes, decisions = paths[context_word]
          @views @inbounds results = σ.(output[nodes, :])
          @views δs = ((1 .- results) .* (1 .- 2 .* decisions))'
          @views @inbounds mul!(∇out[:, nodes], latent, δs, 1, 1)
          @views @inbounds ∇h .= dropdims(sum(reshape(δs, length(nodes), 1, num_senses) .* reshape(model.out[:, nodes], length(nodes), num_dims, 1), dims=1), dims=1) .* sense_likelihoods[:, j]'
          @views @inbounds ∇in_senses[:, :, word] .+= ∇h
          @views @inbounds ∇in_subwords[:, subwords] .+= sum(∇h, dims=2)
          L += AdaSubGram.HuffmanTree.hierarchical_softmax_loss(results, decisions)
        end
      end
      scaling_factor = length(training_data) / length(minibatch) / 40000
      L += λ / scaling_factor * (norm(model.out) + norm(model.in_senses) + norm(model.in_subwords))
      η = 0.025 * (1 - epoch / epochs)
      ρ = η * scaling_factor
      for (j, (word, _, _)) in enumerate(minibatch)
        @views @inbounds model.ns[:, word] .= (1-η) .* model.ns[:, word] .+ (η*model.word_counts[word]) .* sense_likelihoods[:, j]
      end
      @views model.out .-= ρ .* ∇out .+ λ .* model.out
      @views model.in_senses .-= ρ .* ∇in_senses .+ λ .* model.in_senses
      @views model.in_subwords .-= ρ .* ∇in_subwords .+ λ .* model.in_subwords
    end
    L /= length(training_data)
    finish!(progress)
    println("Total training loss at epoch $epoch/$epochs: $L.")
  end
end

end
