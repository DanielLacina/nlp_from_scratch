use regex::Regex;
use std::collections::HashMap;

pub struct Tokenizer {
   word_index: HashMap<String, i32>, 
   distinct_word_count: i32,
   oov_token: Option<String>
}

impl Tokenizer {
   pub fn new(oov_token: Option<String>) -> Self {
      let mut word_index = HashMap::new(); 
      let mut distinct_word_count = 1;
      if let Some(oov_token) = oov_token.clone() {
        word_index.insert(oov_token, distinct_word_count);
        distinct_word_count += 1; 
      }
      return Self {
        distinct_word_count, 
        oov_token,
        word_index
      } 
   }

   fn extract_words(&self, sentence: &str) -> Vec<String> {
        let re = Regex::new(r"\w+").unwrap();
        let words: Vec<String> = re.find_iter(sentence).map(|m| m.as_str().to_string()).collect();
        return words;
   }
   pub fn fit_on_texts(&mut self, sentences: &Vec<String>) {
        for sentence in sentences {
            let words = self.extract_words(sentence);
            for word in &words {
                if !self.word_index.contains_key(word) {
                    self.distinct_word_count += 1;
                    self.word_index.insert(word.clone(), self.distinct_word_count);
                }  
            }
        }
    }   

    pub fn texts_to_sequences(&self, sentences: &Vec<String>) -> Vec<Vec<i32>> {
        let mut sequences = Vec::new(); 
        let mut max_sequence_length = 0; 
        for sentence in sentences {
             let mut sequence = Vec::new();
             let words = self.extract_words(sentence) ;
            
             for word in words {
                 if let Some(index) = self.word_index.get(&word) {
                    sequence.push(*index);
                 } else if let Some(oov_token) = self.oov_token.as_ref() {
                    if let Some(index) = self.word_index.get(oov_token) {
                        sequence.push(*index);
                    }     
                 }    
             }
             if sequence.len() > max_sequence_length {
                max_sequence_length = sequence.len();
             }
             sequences.push(sequence);
        }
        for sequence in sequences.iter_mut() {
            let padding_needed =  max_sequence_length - sequence.len();
            for _ in 0..padding_needed {
                sequence.push(0);
            }
        }
        return sequences;
    }

    pub fn word_index(&self) -> &HashMap<String, i32> {
        return &self.word_index;
    }
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test] 
   fn test_tokenize_text() {
    let sentences = vec!["hi my name is bob".to_string(), "hi my name is jack lol".to_string()];
    let mut tokenizer = Tokenizer::new(Some("<OOP>".to_string()));
    tokenizer.fit_on_texts(&sentences);
    let sequences = tokenizer.texts_to_sequences(&sentences);
    println!("{:?}", sequences);
   }
} 
