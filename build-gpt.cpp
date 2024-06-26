// torch-apps.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <ATen/ATen.h>
#include <iostream>

// download shakespeare.txt from https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
// and place it in the same directory as this file
#include <fstream>
#include <streambuf>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <map>
#include <torch/torch.h>

#define VOCAB_DATATYPE torch::kInt32

// Hyperparameters
const int block_size = 128; // number of characters in a sequence
const int batch_size = 16; // number of sequences in a batch
const int max_iters = 50000; // number of training iterations
const int eval_interval = 1000; // number of iterations between evaluations
const int eval_iters = 1000; // number of iterations to evaluate the model
const int n_embd = 256; // number of embeddings in the model
const int n_head = 8; // number of attention heads
const int n_layer = 6; // number of transformer layers
const int head_size = 32; // size of each attention head
const int ff_hidden_size = 1024; // size of the hidden layer in the feedforward network
const float dropout = 0.1; // dropout probability
const float learning_rate = 0.0001; // learning rate for the optimizer


std::map<char, int> uniqueCharIndices(const std::string& s);
std::map<int, char> indexToChar(const std::map<char, int>& ctoi);
std::pair<torch::Tensor, torch::Tensor> createInputTargetSequences(const torch::Tensor& encoded_text);
std::vector<int> tensor_to_vector(const torch::Tensor& tensor);

int main() {

	// ------------------- Reading the file -------------------
	std::ifstream t("shakespeare.txt");
	std::string shakespeare_txt((std::istreambuf_iterator<char>(t)),
		std::istreambuf_iterator<char>());

	//std::cout << str.substr(0, 100) << std::endl;
	//std::cout << "Length of string: " << str.length() << std::endl;

	std::map<char, int> ctoi = uniqueCharIndices(shakespeare_txt);
	std::map<int, char> itoc = indexToChar(ctoi);

	auto encode = [&ctoi](const std::string& s) {
		std::vector<int> indices;
		for (char c : s) {
			indices.push_back(ctoi[c]);
		}
		return indices;
	};
	
	auto decode = [&itoc](const std::vector<int>& encoded_txt) {
		std::string decoded;
		for (int i : encoded_txt) {
			decoded.push_back(itoc[i]);
		}
		return decoded;
	};

	// ------------------- Split the data -------------------
	
	// encode the entire string
	std::vector<int> data = encode(shakespeare_txt);


	// Split the data into training and validation sets
	int split = 0.8 * shakespeare_txt.length();
	std::vector<int> train_data(data.begin(), data.begin() + split);
	std::vector<int> val_data(data.begin() + split, data.end());

    // convert vectors into torch tensors
	auto train_data_tensor = torch::from_blob(train_data.data(), { (int64_t)train_data.size() }, VOCAB_DATATYPE);
	auto val_data_tensor = torch::from_blob(val_data.data(), { (int64_t)val_data.size() }, VOCAB_DATATYPE);

	// get a batch of input and target sequences
	//auto [input_sequences, target_sequences] = createInputTargetSequences(train_data_tensor);
	// print the first input and target sequences
	//std::cout << "Input sequence: " << decode(tensor_to_vector(input_sequences[0])) << std::endl;
	//std::cout << "Target sequence: " << decode(tensor_to_vector(target_sequences[0])) << std::endl;
	
	// ------------------- Create the model -------------------


	return 0;
}

// create a function that takes a tensor of encoded text and returns a tensor a batch of input and target sequences
std::pair<torch::Tensor, torch::Tensor> createInputTargetSequences(const torch::Tensor& encoded_text) {
	// Get the length of the encoded text
	int n = encoded_text.size(0);

	// Calculate the number of sequences
	int num_sequences = n - block_size;

	// Create tensors to store the input and target sequences
	torch::Tensor input_sequences = torch::zeros({ batch_size, block_size }, VOCAB_DATATYPE);
	torch::Tensor target_sequences = torch::zeros({ batch_size, block_size }, VOCAB_DATATYPE);

	// selected random indices for the sequences
	torch::Tensor indices = torch::randint(0, n - block_size, { batch_size });

	// Iterate over the indices and create the input and target sequences
	for (int i = 0; i < batch_size; ++i) {
		// Get the index
		int idx = indices[i].item<int>();

		// Get the input sequence
		input_sequences[i] = encoded_text.slice(0, idx, idx + block_size);

		// Get the target sequence
		target_sequences[i] = encoded_text.slice(0, idx + 1, idx + block_size + 1);
	}

	return { input_sequences, target_sequences };
}



// Function to convert a torch tensor to a std::vector<float>
std::vector<int> tensor_to_vector(const torch::Tensor& tensor) {
	// Ensure the tensor is on CPU and is contiguous
	torch::Tensor cpu_tensor = tensor.to(torch::kCPU).contiguous();

	// Get the pointer to the tensor's data
	int* data_ptr = cpu_tensor.data_ptr<int32_t>();

	// Create a vector from the tensor data
	std::vector<int> vec(data_ptr, data_ptr + tensor.numel());

	return vec;
}




std::map<char, int> uniqueCharIndices(const std::string& s) {
	// Vector to store the unique characters
	std::vector<char> chars;

	// Iterate through the string and keep track of seen characters
	for (char c : s) {
		// If the character is not already in the vector, add it
		if (std::find(chars.begin(), chars.end(), c) == chars.end()) {
			chars.push_back(c);
		}
	}

	// Sort the characters
	std::sort(chars.begin(), chars.end());

	// Map to store the characters and their indices
	std::map<char, int> charIndices;

	// Iterate through the vector and store the character and its index
	for (int i = 0; i < chars.size(); ++i) {
		charIndices[chars[i]] = i;
	}

	return charIndices;
}

std::map<int, char> indexToChar(const std::map<char, int>& ctoi) {
	std::map<int, char> itoc;
	for (const auto& pair : ctoi) {
		itoc[pair.second] = pair.first;
	}
	return itoc;
}