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

function forward_pass!(model::Parameters, input_word::UInt64, input_subwords::Vector{UInt32}, latent::T, output::T, idx::I) where {T <: AbstractArray{Float32, 2}, I <: AbstractArray{Int32, 1}}
  @views @inbounds sum!(latent[:, 1], model.in_subwords[:, input_subwords])
  @views @inbounds latent[:, 2:end] .= latent[:, 1]
  @views @inbounds add!(latent, model.in_senses[:, :, input_word])
  @views @inbounds mul!(output[:, idx], latent', model.out[:, idx])
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

# TODO set up memory pool struct
# TODO compute loss during sense likelihoods
# TODO subsampling of frequent words
# TODO use StrideArrays.jl
# TODO loss function for everything, now the loss is running, not perfect.
# TODO adam for the gradient updates
# TODO improve memory locality Vector{Tuple} -> Tuple{Vector}
# TODO small Vectors -> SVector (using StaticArrays, for the sense-based likelihoods array for example)
# TODO check types using @code_warntype
# TODO checkout Base.Cartesian.@nexprs
# TODO use similar(array) for array creation
# TODO avoid Dict{Integer, T} for Vector{T}
# TODO investigate using LoopVectorization, Tullio
# TODO investigate usage of Ref{T}
# TODO investigate usage of NTuple{N, T}
# TODO CHECK EVERYTHING!!!! NOTE seems to outperform SkipGram and AdaGram now!

function compute_beta_parameters!(αs::T, βs::T, ns::T, α::Float32) where T <: AbstractArray{Float32, 1}
  fill!(αs, 1.0f0)
  fill!(βs, α)
  cumulative_ns = 0.0f0
  αs .+= ns
  for sense in length(ns):-1:1
    @inbounds cumulative_ns += ns[sense]
    @inbounds βs[sense] += cumulative_ns
  end
  return αs, βs
end

@inline function σ(x::Float32)::Float32
  return 1.0f0 / (1.0f0 + exp(-x))
end

@inline function sigmoid!(arr::A) where A <: AbstractArray{Float32, 2}
  for w in axes(arr, 2)
    @simd ivdep for v in axes(arr, 1)
      @inbounds arr[v, w] = σ(arr[v, w])
    end
  end
end

@inline function add!(to::A, from::B) where {A <: AbstractArray{Float32, 2}, B <: AbstractArray{Float32, 2}}
  for w in axes(to, 2)
    @simd ivdep for v in axes(to, 1)
      @inbounds to[v, w] += from[v, w]
    end
  end
end

@inline function add_all!(to::A, from::B) where {A <: AbstractArray{Float32, 2}, B <: AbstractArray{Float32, 1}}
  for w in axes(to, 2)
    @simd ivdep for v in axes(to, 1)
      @inbounds to[v, w] += from[v]
    end
  end
end

@inline function fastsum_likelihoods!(accs::T, vs::A, decisions::D)::T where {T <: AbstractArray{Float32, 1}, D <: AbstractArray{Float32, 1}, A <: AbstractArray{Float32, 2}}
  for w in axes(vs, 2)
    @simd ivdep for v in axes(vs, 1)
      @inbounds accs[v] += log(vs[v, w] * decisions[w] + (1.0f0 - vs[v, w]) * (1.0f0 - decisions[w]))
    end
  end
  return accs
end

# TODO write some tests for this lmao
# TODO as, bs, sense_likelihoods, bϝs as MVector ?
function sense_likelihoods!(sense_likelihoods::T, as::T, bs::T, output::O, context::S, paths::Vector{Tuple{Vector{Int32}, Vector{Float32}}}, ns::T, bϝs::T, num_senses::Int64, α::Float32) where T <: AbstractArray{Float32, 1} where O <: AbstractArray{Float32, 2} where S <: AbstractArray{UInt64, 1}
  @inbounds @views compute_beta_parameters!(as, bs, ns, α)
  for sense in 1:num_senses
    @inbounds ϝs = digamma(as[sense] + bs[sense])
    @inbounds bϝs[sense] = digamma(bs[sense]) - ϝs
    @inbounds logbeta_k = digamma(as[sense]) - ϝs
    @views @inbounds logbeta_complements = sum(bϝs[1:(sense-1)])
    @inbounds sense_likelihoods[sense] = logbeta_k + logbeta_complements
  end
  @simd for context_word in context
    @inbounds nodes, decisions = paths[context_word]
    @views @inbounds fastsum_likelihoods!(sense_likelihoods, output[:, nodes], decisions)
  end
  max_likelihood = maximum(sense_likelihoods)
  sense_likelihoods .= exp.(sense_likelihoods .- max_likelihood)
  sum_factor = 1.0f0 / sum(sense_likelihoods)
  sense_likelihoods .*= sum_factor
end

struct TrainSettings
  α::Float32
  epochs::Int64
  η_1::Float32
  η_2::Float32
end

function settings(α::Float32=0.1f0, epochs::Int64=5, η_1::Float32=0.025f0, η_2::Float32=NaN32)
  return TrainSettings(α, epochs, η_1, isnan(η_2) ? 2.0f0 * η_1 : η_2)
end

# TODO split up train
function train(model::Parameters, training_data::Vector{Tuple{UInt64, Vector{UInt32}, Vector{UInt64}}}, paths::Vector{Tuple{Vector{Int32}, Vector{Float32}}}, settings::TrainSettings, max_nodes::Int64)
  num_dims, num_senses, _ = size(model.in_senses)
  num_out = size(model.out, 2)
  # TODO move all these preallocated structures into a struct
  ζs = zeros(Float32, num_senses, max_nodes, nthreads())
  scratch_out = zeros(Float32, num_dims, max_nodes, nthreads())
  sense_sums = zeros(Float32, num_senses, nthreads())
  bϝs = zeros(Float32, num_senses, nthreads())
  ∇h = zeros(Float32, num_dims, num_senses, nthreads())
  ∇h_sum = zeros(Float32, num_dims, nthreads())
  latent = zeros(Float32, num_dims, num_senses, nthreads())
  output = zeros(Float32, num_senses, num_out, nthreads())
  sense_likelihoods = zeros(Float32, num_senses, nthreads())
  as = zeros(Float32, num_senses, nthreads())
  bs = zeros(Float32, num_senses, nthreads())
  nodeset = Vector{Set{Int32}}(undef, nthreads())
  for i in eachindex(nodeset)
    @inbounds nodeset[i] = Set{Int32}()
  end
  println("Start at ", now())
  L = 0.0
  for epoch in 0:(settings.epochs-1)
    L = 0.0
    dataset = AdaSubGram.Dataset.shuffle!(training_data)
    @threads for (j, (word, subwords, context)) in dataset
      tid = threadid()
      @inbounds empty!(nodeset[tid])
      for i in eachindex(context)
        @inbounds nodes, _ = paths[context[i]]
        @inbounds union!(nodeset[tid], nodes)
      end
      @inbounds nodevec = collect(nodeset[tid]) # TODO move this stuff to dataset creation! should boost performance by ~2%
      @inbounds @views forward_pass!(model, word, subwords, latent[:, :, tid], output[:, :, tid], nodevec)
      @inbounds @views clamp!(output[:, nodevec, tid], -16.0f0, 16.0f0)
      @inbounds @views sigmoid!(output[:, nodevec, tid])
      @inbounds @views sense_likelihoods!(sense_likelihoods[:, tid], as[:, tid], bs[:, tid], output[:, :, tid], context, paths, model.ns[:, word], bϝs[:, tid], num_senses, settings.α)
      η_1 = settings.η_1 * (1.0f0 - (epoch + j / length(training_data)) / settings.epochs)
      η_2 = settings.η_2 * (1.0f0 - (epoch + j / length(training_data)) / settings.epochs)
      ℓ = 0.0
      for i in eachindex(context)
        @inbounds nodes, decisions = paths[context[i]]
        @views @inbounds ζs[:, 1:length(nodes), tid] .= (decisions' .- output[:, nodes, tid]) .* sense_likelihoods[:, tid]
        @views @inbounds mul!(∇h[:, :, tid], scratch_out[:, 1:length(nodes), tid], ζs[:, 1:length(nodes), tid]', η_1 / (1 + length(subwords)), 0.0f0)
        @views @inbounds scratch_out[:, 1:length(nodes), tid] .= model.out[:, nodes]
        @views @inbounds mul!(scratch_out[:, 1:length(nodes), tid], latent[:, :, tid], ζs[:, 1:length(nodes), tid], η_1, 1.0f0)
        @views @inbounds model.out[:, nodes] .= scratch_out[:, 1:length(nodes), tid]
        if epoch % 2 == 1 || settings.epochs < 2
          @views @inbounds add!(model.in_senses[:, :, word], ∇h[:, :, tid])
        end
        if epoch % 2 == 0
          @views @inbounds sum!(∇h_sum[:, tid], ∇h[:, :, tid])
          @views @inbounds add_all!(model.in_subwords[:, subwords], ∇h_sum[:, tid])
        end
        @views @inbounds ℓ += AdaSubGram.HuffmanTree.hierarchical_softmax_loss(output[:, nodes, tid], decisions, sense_likelihoods[:, tid], sense_sums[:, tid])
      end
      L += ℓ / length(context)
      @views @inbounds model.ns[:, word] .= (1.0f0 - η_2) .* model.ns[:, word] .+ η_2 .* model.word_counts[word] .* sense_likelihoods[:, tid]
    end
    L /= length(training_data)
    println("Total training loss at epoch ", epoch+1, "/", settings.epochs, ": ", L, " at ", now())
  end
  return L
end

end
