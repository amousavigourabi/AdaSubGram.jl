module Parser

const Filepath=String

"""
    parse(input::Filepath) -> String

Parses transliterated OCS texts from the corpus.
Note: incredibly naive implementation.
"""
function parse(input::Filepath)::String
  file_contents = readlines(input)
  clear_starts = [replace(file_content, r"\d+ " => "") for file_content in file_contents]
  clear_punctuations = [replace(clear_start, r"[\.\n\-!*'~^(:\[\]\?\{\}=<>/\\]" => "") for clear_start in clear_starts]
  collapse_whitespaces = [strip(replace(clear_punctuation, r"[ ]+" => " ")) for clear_punctuation in clear_punctuations]
  merged_yers = [replace(collapse_whitespace, r"&" => "\$") for collapse_whitespace in collapse_whitespaces]
  document = join(merged_yers, " ")
  return document
end

export parse

end
