module Model

using AdaSubGram
using LinearAlgebra
using SpecialFunctions
using ProgressMeter

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
  senses::Int64
  word_counts::Vector{UInt64}
end

function initialize(vector_dims::Int64, word_counts::Vector{UInt64}, input_subwords::Int64, senses::Int64)
  input_words = length(word_counts)
  in_subwords = (rand(Float32, input_subwords, vector_dims) .- 0.5) ./ vector_dims
  in_senses = (rand(Float32, input_words, vector_dims, senses) .- 0.5) ./ vector_dims
  out = (rand(Float32, vector_dims, input_words - 1) .- 0.5) ./ vector_dims
  ns = zeros(Float32, input_words, senses)
  @views ns[:, 1] .+= word_counts
  model = Parameters(in_subwords, in_senses, out, ns, senses, word_counts)
  return model
end

function forward_pass(model::Parameters, input_word::UInt64, input_subwords::Vector{UInt32})::Tuple{Array{Float32, 2}, Array{Float32, 2}}
  @views latent_representation = model.in_senses[input_word, :, :]' .+ sum(model.in_subwords[input_subwords, :], dims=1)
  scale_latent = mapreduce(abs, max, latent_representation)
  scale_out = mapreduce(abs, max, model.out)
  latent_representation /= scale_latent
  norm_model_out = model.out / scale_out
  output = scale_latent * scale_out * latent_representation * norm_model_out
  if any(isnan.(output)) || any(isinf, output) # DEBUG
    if any(isnan, output)
      println("output:nan")
    end
    if any(isinf, output)
      println("output:inf")
    end
    if any(isnan, latent_representation)
      println("latent_representation:nan")
    end
    if any(isinf, latent_representation)
      println("latent_representation:inf")
    end
    if any(isnan, model.out)
      println("model.out:nan")
    end
    if any(isinf, model.out)
      println("model.out:inf")
    end
    if any(isnan, model.in_senses[input_word, :, :])
      println("model.in_senses[input_word, :, :]:nan")
    end
    if any(isinf, model.in_senses[input_word, :, :])
      println("model.in_senses[input_word, :, :]:inf")
    end
    if any(isnan, sum(model.in_subwords[input_subwords, :], dims=1))
      println("sum(model.in_subwords[input_subwords, :], dims=1):nan")
    end
    if any(isinf, sum(model.in_subwords[input_subwords, :], dims=1))
      println("sum(model.in_subwords[input_subwords, :], dims=1):inf")
    end
  end
  return latent_representation, output
end

# function sense_likelihoods(model::Parameters)::Array{Float32, 2}
#   words, senses = size(model.ns) # todo deal with ns instead of as, bs
#   likelihoods = zeros(Float32, words, senses)
#   for sense in 1:senses
#     likelihoods[:, sense] = as[:, sense] ./ (as[:, sense] .+ bs[:, sense])
#   end
#   prod = ones(Float32, words)
#   for sense in 1:(senses-1)
#     prod *= bs[:, sense] ./ (as[:, sense] .+ bs[:, sense])
#     likelihoods[:, sense+1] *= prod
#   end
#   return likelihoods
# end

function compute_beta_parameters(ns::Vector{Float32}, α::Float32)
  as = ones(Float32, length(ns))
  bs = fill(α, length(ns))
  cumulative_ns = zeros(Float32, length(ns))
  for (i, n) in reverse(collect(enumerate(ns)))
    cumulative_ns[i] = cumulative_ns[min(i, end)] + n
    as[i] += n
    bs[i] += cumulative_ns[i]
  end
  return as, bs
end

function σ(x::Float32)
  return 1 / (1 + exp(-x))
end

# TODO split up train
function train(model::Parameters, training_data::Vector{Tuple{UInt64, Vector{UInt32}, Vector{UInt64}}}, paths::Vector{Tuple{Vector{Int32}, Vector{Float32}}}, batch_size::Int64, epochs::Int64, α::Float32, λ::Float32)
  for epoch in 1:epochs
    L = 0
    minibatches = AdaSubGram.Dataset.minibatches(training_data, batch_size)
    progress = Progress(length(minibatches), desc="Batches epoch $epoch/$epochs")
    ϝs = zeros(Float32, model.senses)
    bϝs = zeros(Float32, model.senses)
    for minibatch in minibatches
      next!(progress)
      sense_likelihoods = zeros(Float32, length(minibatch), model.senses)
      ∇out = zeros(Float32, size(model.out))
      ∇h = zeros(Float32, size(model.in_senses)[2], model.senses)
      ∇in_senses = zeros(Float32, size(model.in_senses))
      ∇in_subwords = zeros(Float32, size(model.in_subwords))
      for (j, (word, subwords, context)) in enumerate(minibatch)
        latent, output = forward_pass(model, word, subwords)
        as, bs = compute_beta_parameters(model.ns[word, :], α)
        for sense in 1:model.senses
          ϝs[sense] = digamma(as[sense] + bs[sense])
          bϝs[sense] = digamma(bs[sense]) - ϝs[sense]
          logbeta_k = digamma(as[sense]) - ϝs[sense]
          logbeta_complements = sum([bϝs[r] for r in 1:(sense-1)])
          sense_likelihoods[j, sense] = logbeta_k + logbeta_complements
        end
        for context_word in context
          for n in paths[context_word][1]
            @views sense_likelihoods[j, :] .+= output[:, n] # DOES NOT INTRODUCE NEW NANS
          end
        end
        pre_DEBUG = deepcopy(sense_likelihoods) # DEBUG
        clamp!(sense_likelihoods, -80, 80)
        clamp_DEBUG = deepcopy(sense_likelihoods) # DEBUG
        sense_likelihoods .= exp.(sense_likelihoods)  # DOES NOT INTRODUCE NEW NANS
        exp_DEBUG = deepcopy(sense_likelihoods) # DEBUG
        totals = sum(sense_likelihoods, dims=2)
        sense_likelihoods ./= totals                 # DOES NOT INTRODUCE NEW NANS
        if any(isnan.(sense_likelihoods)) # DEBUG
          println("oops")
          if any(isnan.(pre_DEBUG))
            println("pre_DEBUG")
          end
          if any(isnan.(clamp_DEBUG))
            println("clamp_DEBUG")
          end
          if any(isnan.(exp_DEBUG))
            println("exp_DEBUG")
          end
          if any(isnan.(totals))
            println("totals")
          end
          if any(isnan.(model.out))
            println("model.out")
          end
          if any(isnan.(model.in_senses))
            println("model.in_senses")
          end
          if any(isnan.(model.in_subwords))
            println("model.in_subwords")
          end
          if any(isnan.(output))
            println("output")
          end
          exit(1)
        end
        for sense in 1:model.senses
          for context_word in context
            # TODO fix these gradients
            for (n, d) in zip(paths[context_word]...)
              @views z = model.out[:, n] .* latent[sense, :]
              p = σ.(z)
              δ = p .- d
              @views ∇out[:, n] .+= δ .* latent[sense, :]
              @views ∇h[:, sense] .+= δ .* model.out[:, n]
            end
            @views ∇in_senses[word, :, sense] .+= sense_likelihoods[j, sense] .* ∇h[:, sense]
            @views ∇in_subwords[subwords, :] .+= sense_likelihoods[j, sense] .* reshape(∇h[:, sense], 1, :)
            @views results = σ.(output[sense, paths[context_word][1]])
            targets = paths[context_word][2]
            L += AdaSubGram.HuffmanTree.hierarchical_softmax_loss(results, targets)
          end
        end
        @views ∇in_senses .*= reshape(sense_likelihoods[j, :], 1, 1, :)
      end
      scaling_factor = length(training_data) / length(minibatch)
      L += λ / scaling_factor * (norm(model.out) + norm(model.in_senses) + norm(model.in_subwords))
      η = 0.025 * (1 - epoch / epochs)
      ρ = η * scaling_factor
      for (j, (word, _, _)) in enumerate(minibatch)
        @views model.ns[word, :] .= (1-η) .* model.ns[word, :] .+ (η*model.word_counts[word]) .* sense_likelihoods[j, :]
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
