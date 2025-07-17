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

function forward_pass!(model::Parameters, input_word::UInt64, input_subwords::Vector{UInt32}, latent::T, output::T)::Nothing where T <: AbstractArray{Float32, 2}
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

# TODO use StrideArrays.jl
# TODO improve memory locality Vector{Tuple} -> Tuple{Vector}
# TODO small Vector -> SVector (using StaticArrays, for the sense-based likelihoods array for example)
# TODO check if we can use @fastmath anywhere
# TODO check types using @code_warntype
# TODO checkout Base.Cartesian.@nexprs
# TODO use similar(array) for array creation
# TODO avoid Dict{Integer, T} for Vector{T}
# TODO investigate using LoopVectorization.jl
# TODO investigate usage of Ref{T}
# TODO investigate usage of NTuple{N, T}

# TODO review
function compute_beta_parameters(ns::T, α::Float32) where T <: AbstractArray{Float32, 1}
  sense = length(ns)
  as = ones(Float32, sense)
  bs = fill(α, sense)
  cumulative_ns = zeros(Float32, sense)
  for n in reverse(ns)
    @views @inbounds cumulative_ns[1:sense] .+= n
    @inbounds as[sense] += n
    @inbounds bs[sense] += cumulative_ns[sense]
    sense -= 1
  end
  return as, bs
end

function σ(x::Float32)
  return 1 / (1 + exp(-x))
end

function logσ(x::Float32)
  return log(1 / (1 + exp(-x)))
end

# TODO fix sense_likelihoods!
# TODO write some tests for it lmao
function sense_likelihoods!(sense_likelihoods::T, output::O, context::S, paths::Vector{Tuple{Vector{Int32}, Vector{Float32}}}, ns::T, bϝs::T, num_senses::Int64, α::Float32)::Nothing where T <: AbstractArray{Float32, 1} where O <: AbstractArray{Float32, 2} where S <: AbstractArray{UInt64, 1}
  @inbounds @views as, bs = compute_beta_parameters(ns, α)
  for sense in 1:num_senses
    @inbounds ϝs = digamma(as[sense] + bs[sense])
    @inbounds bϝs[sense] = digamma(bs[sense]) - ϝs
    @inbounds logbeta_k = digamma(as[sense]) - ϝs
    @views @inbounds logbeta_complements = sum(bϝs[1:(sense-1)])
    @inbounds sense_likelihoods[sense] = logbeta_k + logbeta_complements
  end
  @simd for context_word in context
    @inbounds nodes, decisions = paths[context_word]
    @views @inbounds sense_likelihoods .+= sum(logσ.(output[nodes, :] .* (1 .- 2 .* decisions)), dims=1)' # allocations
  end
  max_likelihood = maximum(sense_likelihoods)
  @views @inbounds sense_likelihoods .= exp.(sense_likelihoods .- max_likelihood) ./ sum(exp.(sense_likelihoods .- max_likelihood)) # allocations
  return nothing
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
  println("Start at ", now())
  for epoch in 0:(epochs-1)
    L = 0.0
    dataset = AdaSubGram.Dataset.shuffle!(training_data)
    @threads for (j, (word, subwords, context)) in dataset # TODO @threads calibration is poor at the start?
      @inbounds @views forward_pass!(model, word, subwords, latent[:, :, threadid()], output[:, :, threadid()]) # allocations
      @inbounds @views clamp!(output[:, :, threadid()], -80.0f0, 80.0f0)
      @inbounds @views sense_likelihoods!(sense_likelihoods[:, threadid()], output[:, :, threadid()], context, paths, model.ns[:, word], bϝs[:, threadid()], num_senses, α)
      η = 0.0025f0 * (1 - (epoch + (j - 1) / length(training_data)) / epochs)
      @simd for context_word in context
        # TODO check gradient updates!
        @inbounds nodes, decisions = paths[context_word]
        @views @inbounds results = σ.(output[nodes, :, threadid()]) # allocations
        @views @inbounds ζs = (decisions .- results) .* sense_likelihoods[:, threadid()]' # allocations
        @views @inbounds mul!(model.out[:, nodes], latent[:, :, threadid()], ζs', η, 1.0f0)
        @views @inbounds mul!(∇h[:, :, threadid()], model.out[:, nodes], ζs)
        @views @inbounds model.in_senses[:, :, word] .+= η .* ∇h[:, :, threadid()]
        @views @inbounds model.in_subwords[:, subwords] .+= η .* sum(∇h[:, :, threadid()], dims=2) # allocations
        @views @inbounds L += AdaSubGram.HuffmanTree.hierarchical_softmax_loss(results, decisions, sense_likelihoods[:, threadid()])
      end
      # if isnan(maximum(model.out))
      #   println("Hit a NaN at ", now())
      #   exit(1)
      # end
      @views @inbounds model.ns[:, word] .= (1 - η) .* model.ns[:, word] .+ η .* model.word_counts[word] .* sense_likelihoods[:, threadid()]
    end
    L /= length(training_data)
    println("Total training loss at epoch ", epoch+1, "/", epochs, ": ", L, " at ", now())
  end
end

end
