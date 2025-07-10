module Model

using AdaSubGram
using LinearAlgebra
using SpecialFunctions
using Base.Threads
using Dates

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
function train(model::Parameters, training_data::Vector{Tuple{UInt64, Vector{UInt32}, Vector{UInt64}}}, paths::Vector{Tuple{Vector{Int32}, Vector{Float32}}}, epochs::Int64, α::Float32)
  num_dims, num_senses, _ = size(model.in_senses)
  _, num_out = size(model.out)
  bϝs = zeros(Float32, num_senses, nthreads())
  ∇h = zeros(Float32, num_dims, num_senses, nthreads())
  latent = zeros(Float32, num_dims, num_senses, nthreads())
  output = zeros(Float32, num_out, num_senses, nthreads())
  sense_likelihoods = zeros(Float32, num_senses, nthreads())
  println("Start at $(now())")
  for epoch in 0:(epochs-1)
    L = 0
    dataset = AdaSubGram.Dataset.shuffle!(training_data)
    @threads for (j, (word, subwords, context)) in dataset # @threads calibration is poor at the start ?
      @inbounds @views forward_pass!(model, word, subwords, latent[:, :, threadid()], output[:, :, threadid()]) # allocations
      @inbounds @views as, bs = compute_beta_parameters(model.ns[:, word], α)
      for sense in 1:num_senses
        @inbounds ϝs = digamma(as[sense] + bs[sense])
        @inbounds bϝs[sense, threadid()] = digamma(bs[sense]) - ϝs
        @inbounds logbeta_k = digamma(as[sense]) - ϝs
        @views @inbounds logbeta_complements = sum(bϝs[1:(sense-1), threadid()])
        @inbounds sense_likelihoods[sense, threadid()] = logbeta_k + logbeta_complements
      end
      @simd for context_word in context
        @inbounds nodes, _ = paths[context_word]
        @views @inbounds sense_likelihoods[:, threadid()] .+= sum(output[nodes, :, threadid()])
        @views @inbounds max_outputs = maximum(output[nodes, :, threadid()], dims=1) # allocations
        @views @inbounds sense_likelihoods[:, threadid()] .-= sum(max_outputs .+ log.(sum(exp.(output[nodes, :, threadid()] .- max_outputs), dims=1))) # allocations
      end
      @views @inbounds max_sense = maximum(sense_likelihoods[:, threadid()])
      @views @inbounds sense_likelihoods[:, threadid()] .-= max_sense + log(sum(exp.(sense_likelihoods[:, threadid()] .- max_sense)))
      @views @inbounds sense_likelihoods[:, threadid()] .= exp.(sense_likelihoods[:, threadid()])
      η = 0.025 * (1 - (epoch + (j - 1) / length(training_data)) / epochs)
      @simd for context_word in context
        # TODO check gradient updates!
        @inbounds nodes, decisions = paths[context_word]
        δs = (1 .- 2 .* decisions)' # allocations
        @views @inbounds results = σ.(output[nodes, :, threadid()] .* δs') # allocations
        ζs = (1 .- results) .* δs' # allocations
        @views @inbounds mul!(model.out[:, nodes], latent[:, :, threadid()], ζs', -η, 1)
        @views @inbounds ∇h[:, :, threadid()] .= model.out[:, nodes] * ζs .* sense_likelihoods[:, threadid()]' # allocations
        @views @inbounds model.in_senses[:, :, word] .-= η .* ∇h[:, :, threadid()]
        @views @inbounds model.in_subwords[:, subwords] .-= η .* sum(∇h[:, :, threadid()], dims=2) # allocations
        L += AdaSubGram.HuffmanTree.hierarchical_softmax_loss(results)
      end
      @views @inbounds model.ns[:, word] .= (1 - η) .* model.ns[:, word] .+ η .* model.word_counts[word] .* sense_likelihoods[:, threadid()]
    end
    L /= length(training_data)
    println("Total training loss at epoch $(epoch+1)/$epochs: $L at $(now()).")
  end
end

end
