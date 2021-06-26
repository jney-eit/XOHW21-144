#include "host.hpp"




std::vector<std::string> open(std::string path)
{
	DIR*    dir;
	dirent* pdir;
	std::vector<std::string> files;

	dir = opendir(path.empty() ? "." : path.c_str());

	while ((pdir = readdir(dir))) {
		files.push_back(pdir->d_name);
	}

	closedir(dir);

	std::sort(files.begin(), files.end());
	files.erase(files.begin(), files.begin()+2);

	return files;
}



std::vector<ap_uint<8>> ReadBinaryFile(std::string path){
	std::ifstream input_stream;
	input_stream.open(path, std::ifstream::in | std::ios_base::binary);

	// Read Header
	for(unsigned int i = 0; i < 16; i++){
		std::string line;
		getline(input_stream, line);
	}
	
	// Get first "00" byte
	unsigned char temp;
	input_stream.read((char *) &temp, 1);
	while(temp != 0){
		input_stream.read((char *) &temp, 1);
	}

	std::vector<ap_uint<8>> data(SEQUENCE_LENGTH);

	unsigned int skip_count = 0;
	unsigned int channel_count = 0;
	unsigned int data_iter = 0;

	for(int i = 0; i < SEQUENCE_LENGTH * INPUT_VAL_SKIP_FAC; i+=2){
		if(skip_count == 0){

			ap_uint<8> in_0;
			ap_uint<8> in_1;
			ap_uint<8> in_2;
			ap_uint<8> in_3 ;
			
			input_stream.read((char*) &in_0, 1);
			input_stream.read((char*) &in_1, 1);
			input_stream.read((char*) &in_2, 1);
			input_stream.read((char*) &in_3, 1);
			
			data[data_iter](3, 0) = in_3(3, 0);
			data[data_iter](7, 4) = in_2(7, 4);
			data[data_iter + 1](3, 0) = in_1(3, 0);
			data[data_iter + 1](7, 4) = in_0(7, 4);

			data_iter+=2;
		}else{
			ap_uint<32> temp;
			input_stream.read((char *) &temp, 4);
		}
		channel_count+=2;
		if(channel_count == INPUT_CHANNELS){
			channel_count = 0;
			skip_count++;
			if(skip_count == INPUT_VAL_SKIP_FAC){
				skip_count = 0;
			}
		}
	}

	std::cout << "End ReadBinaryFile" << std::endl;
	return data;
}




std::vector<float> ReadImageFromFile(std::string path)
{
	std::vector<float> image;
	std::ifstream input_stream;

	input_stream.open(path, std::ifstream::in);
	if (!input_stream.good()) {
		throw std::runtime_error("No image file found at: " + path);
	}

	float pix;
	unsigned int skip_count = 0;
	unsigned int channel_count = 0;


	while(input_stream >> pix)
	{
		if(skip_count == 0){

			// Resolve wrong normalization factor
			// pix = pix * 1.331;

			image.push_back(pix);
		}
		channel_count++;
		if(channel_count == INPUT_CHANNELS){
			channel_count = 0;
			skip_count++;
			if(skip_count == INPUT_VAL_SKIP_FAC){
				skip_count = 0;
			}
		}
	}
	input_stream.close();

	return image;
}

template <typename T>
InputImage<T>::InputImage()
{
	image_ = NULL;
	number_of_patches = 0;
}

template <typename T>
InputImage<T>::~InputImage()
{
	if (image_ != NULL)
	{
		delete[] image_;
		image_ = NULL;
	}
}

template <typename T>
template
<
unsigned int TransactionWidth,
unsigned int InputLength,
unsigned int InputChannels,
unsigned int InputWidth,
typename Pixel_t
>
void InputImage<T>::InitInputImage(std::vector<float> image)
{

	std::cout << "InitInputImage..." << std::endl;

	if (image.size() % InputChannels != 0)
	{
		throw std::runtime_error("ERROR: The input sequence is not a multiple of number of input channels");
	}

	if (TransactionWidth % InputWidth != 0)
	{
		throw std::runtime_error("ERROR: The transaction width has to be a multiple of an input width");
	}


	constexpr unsigned int inputs_per_transaction = TransactionWidth / InputWidth;


	if ((image.size() / InputChannels) % inputs_per_transaction != 0)
	{
		unsigned int number_values_subtract = image.size() - (image.size() / inputs_per_transaction) * inputs_per_transaction;

		for(unsigned int n = 0; n < number_values_subtract * InputChannels; n++)
			image.erase(image.end());
	}

	std::vector<ap_uint<TransactionWidth> > packed_input;
	unsigned int transactions_per_sequence = image.size() / inputs_per_transaction;

	for(unsigned int tps = 0; tps < transactions_per_sequence; tps++)
	{
		Pixel_t fpix[inputs_per_transaction];

		for (unsigned int ipt = 0; ipt < inputs_per_transaction; ipt++)
		{
			fpix[ipt] = (Pixel_t) image.at(tps * inputs_per_transaction + ipt);
		}

		ap_uint<TransactionWidth> pack = Pack<TransactionWidth, Pixel_t, InputWidth, inputs_per_transaction>(fpix);

		packed_input.push_back(pack);

	}// #transaction

	//std::cout << "Allocating a buffer for packed transactions..." << std::endl;

	if (image_ == NULL){
		image_ = new T[packed_input.size()];
	}else{
		throw std::runtime_error("ERROR: Internal buffer 'image_' has not a NULL pointer");
	}

	for(int word = 0; word < packed_input.size(); word++){
		image_[word] = packed_input.at(word);
	}
}



template
<
unsigned int TransactionWidth
>
std::vector<ap_uint<TransactionWidth> > InitECG(std::vector<ap_uint<16>> ecg_data)
{

	std::cout << "InitECG..." << std::endl;


	constexpr unsigned int inputs_per_transaction = TransactionWidth / INPUT_BITS_PER_VAL;
	constexpr unsigned int input_overhead = TransactionWidth % INPUT_BITS_PER_VAL;


	std::vector<ap_uint<TransactionWidth> > packed_input;
	unsigned int transactions_per_sequence = ecg_data.size() / inputs_per_transaction;

	for(unsigned int tps = 0; tps < transactions_per_sequence; tps++)
	{

			ap_uint<INPUT_BITS_PER_VAL> fpix[inputs_per_transaction];

			for (unsigned int ipt = 0; ipt < inputs_per_transaction; ipt+=2)
			{

				ap_uint<16> temp_0 = ecg_data.at(tps * inputs_per_transaction + ipt);
				ap_uint<16> temp_1 = ecg_data.at(tps * inputs_per_transaction + ipt + 1);
				ap_uint<16> mask = INPUT_MASK;

				unsigned int out_iter = 0;
				for(unsigned int bit_iter = 0; bit_iter < 16; bit_iter++){
					if(mask[bit_iter] == 1){
						fpix[ipt][out_iter] = temp_1[bit_iter];
						fpix[ipt + 1][out_iter] = temp_0[bit_iter];
						out_iter++;
					}
				}
			}

			ap_uint<TransactionWidth> pack = Pack<TransactionWidth, ap_uint<INPUT_BITS_PER_VAL>, INPUT_BITS_PER_VAL, inputs_per_transaction>(fpix);

		packed_input.push_back(pack);

	}

	return packed_input;
}




template <typename T>
void InputImage<T>::InitGTImage(std::vector<float> image)
{
	if (image.size() != 1)
	{
		throw std::runtime_error("ERROR: Incorrect number of values in the ground truth image");
	}

	std::cout << "Allocating a gt buffer ..." << std::endl;

	if (image_ == NULL)
		image_ = new T[image.size()];
	else
	{
		throw std::runtime_error("ERROR: Internal gt buffer 'image_' has not a NULL pointer");
	}

	for(int i = 0; i < image.size(); i++)
	{
		image_[i] = (T)image.at(i);
	}

	std::cout << "Allocation completed." << std::endl;

}

template <typename T>
template
<
unsigned int TransactionWidth,
unsigned int ClassLabelWidth
>
void InputImage<T>::InitPredictedImage(std::vector<float> image)
{
	unsigned int numberWordsPixels = 1;

	std::cout << "Allocating a buffer for prediction..." << std::endl;

	if (image_ == NULL)
		image_ = new T[numberWordsPixels];
	else
	{
		throw std::runtime_error("ERROR: Internal prediction buffer 'image_' has not a NULL pointer");
	}

	for(int i = 0; i < numberWordsPixels; i++)
		image_[i] = (T)0;

	std::cout << "Allocation completed." << std::endl;
}

void arrhythmia_wrapper(unsigned int numberOfInputsToProcess,
						std::string InputPathDir,
						std::string GTPathDir,
						std::vector<unsigned int>& errors,
						float &false_positive_rate,
						float &false_negative_rate)
{
	std::cout << "Preparing inputs ..." << std::endl;

	// Return the list of inputs' file names
	std::vector<std::string> listOfInputs = open(InputPathDir);

	unsigned int number_of_inputs = 0;

	if((numberOfInputsToProcess == 0) || (numberOfInputsToProcess > listOfInputs.size()))
	{
		number_of_inputs = listOfInputs.size();
	}
	else
	{
		number_of_inputs = numberOfInputsToProcess;
	}

	std::cout << "Number of inputs is " << number_of_inputs << std::endl;

#if USE_ECG == 1
	std::vector<std::vector<ap_uint<DATAWIDTH_IN>>> vecInputs;
#else
	std::vector<InputImage<ap_uint<DATAWIDTH_IN>>> vecInputs;
#endif

	vecInputs.resize(number_of_inputs);
	std::sort(listOfInputs.begin(), listOfInputs.end());


	for(unsigned int i = 0; i < number_of_inputs; i++)
	{
		std::string path = InputPathDir + listOfInputs.at(i);
		std::cout << "Current input: " << path << std::endl;

#if USE_ECG == 1

	std::vector<ap_uint<8>> ecg_data = ReadBinaryFile(path);

	unsigned int overhead = (ecg_data.size() * 8) % DATAWIDTH_IN;
	if (overhead){
		unsigned int num_bits_push_back = DATAWIDTH_IN - ((ecg_data.size() * 8) % DATAWIDTH_IN);
		for(int i = 0; i < num_bits_push_back; i+=8){
			ecg_data.push_back(0);
		}
	}

	std::cout << "Setting vec Inputs..." << std::endl;
	vecInputs.at(i) = *reinterpret_cast<std::vector<ap_uint<DATAWIDTH_IN>>*>(&ecg_data);
	std::cout << "Setting vec Inputs done" << std::endl;

#else
	// Assumes first layer is conv, has to be changed if it is not
	typedef ap_fixed<MAXPOOL1D_0_IA_BITS, MAXPOOL1D_0_IA_INT_BITS, AP_RND_ZERO, AP_WRAP> Input_t;

	vecInputs.at(i).InitInputImage<
								   DATAWIDTH_IN,
								   MAXPOOL1D_0_IFM_DIM,
								   MAXPOOL1D_0_IFM_CH,
								   MAXPOOL1D_0_IA_BITS,
								   Input_t
								   >
								   (ReadImageFromFile(path));

#endif
	}
	//=====================================================================================
	//=====================================================================================

	std::cout << "Preparing GT inputs..." << std::endl;

	// Return the list of images' file names
	std::vector<std::string> listOfGTImages = open(GTPathDir);

	if (listOfInputs.size() != listOfGTImages.size())
	{
		throw std::runtime_error("Number of GT files is not equal to a number of Inputs");
	}

	std::vector<InputImage<unsigned int>> vecGTImages;
	vecGTImages.resize(number_of_inputs);

	for(unsigned int i = 0; i < number_of_inputs; i++)
	{
		std::string path = GTPathDir + listOfGTImages.at(i);

		vecGTImages.at(i).InitGTImage(ReadImageFromFile(path));
	}

	//=====================================================================================
	//=====================================================================================

	std::cout << "Allocating memory for predicted images..." << std::endl;

	std::vector<InputImage<ap_uint<DATAWIDTH_OUT>>> vecPredictions;
	vecPredictions.resize(number_of_inputs);

	for(unsigned int i = 0; i < number_of_inputs; i++)
	{
		std::string path = GTPathDir + listOfGTImages.at(i);

		vecPredictions.at(i).InitPredictedImage<
												DATAWIDTH_OUT,
												CLASS_LABEL_BITS
												>
												(ReadImageFromFile(path));
	}

	//=====================================================================================
	//=====================================================================================

	// 'vecInputSinus' contains already a desired number of sinus rhythm based on 'numberOfInputsToProcess'
#if USE_ECG == 1
	arrhythmia_ecg<
			   DATAWIDTH_IN,
			   DATAWIDTH_OUT,
			   CLASS_LABEL_BITS
			  >
			  (vecInputs, vecPredictions);
#else
	arrhythmia<
			   DATAWIDTH_IN,
			   DATAWIDTH_OUT,
			   CLASS_LABEL_BITS
			  >
			  (vecInputs, vecPredictions);
#endif

	//=====================================================================================
	//=====================================================================================

	CalculateAccuracy
	<
	DATAWIDTH_OUT,
	CLASS_LABEL_BITS
	>
	(vecPredictions, vecGTImages, errors, false_positive_rate, false_negative_rate);
}

template
<
unsigned int TRANSACTIONWIDTH_IN,
unsigned int TRANSACTIONWIDTH_OUT,
unsigned int ClassLabelWidth
>
void arrhythmia(std::vector<InputImage<ap_uint<TRANSACTIONWIDTH_IN>>>& vecInputSinus, std::vector<InputImage<ap_uint<TRANSACTIONWIDTH_OUT>>>& vecPrediction)
{
	unsigned int start_sinus = 0;
	unsigned int finish_sinus;

	if(start_sinus == 0)
		finish_sinus = vecInputSinus.size();
	else
		finish_sinus = start_sinus + 1;// selective execution of a single image

	std::cout << "Start processing..." << std::endl;

	typedef ap_fixed<16, 8, AP_RND_ZERO, AP_WRAP> Result_t;

	for(unsigned int sin = start_sinus; sin < finish_sinus; sin++)
	{



		ap_uint<TRANSACTIONWIDTH_OUT> *pPrediction = vecPrediction.at(sin).image_;

		ap_uint<TRANSACTIONWIDTH_IN> *pInputSinus = vecInputSinus.at(sin).image_;
        TopLevel(pInputSinus ,pPrediction);

		std::cout << "Finished calculation for Sinus Nr." << sin + 1 << " of " << finish_sinus << std::endl;
	}
}



template
<
unsigned int TRANSACTIONWIDTH_IN,
unsigned int TRANSACTIONWIDTH_OUT,
unsigned int ClassLabelWidth
>
void arrhythmia_ecg(std::vector<std::vector<ap_uint<TRANSACTIONWIDTH_IN>>>& vecInputSinus, std::vector<InputImage<ap_uint<TRANSACTIONWIDTH_OUT>>>& vecPrediction)
{
	unsigned int start_sinus = 0;
	unsigned int finish_sinus;

	if(start_sinus == 0)
		finish_sinus = vecInputSinus.size();
	else
		finish_sinus = start_sinus + 1;// selective execution of a single image

	std::cout << "Start processing..." << std::endl;

	typedef ap_fixed<16, 8, AP_RND_ZERO, AP_WRAP> Result_t;

	for(unsigned int sin = start_sinus; sin < finish_sinus; sin++)
	{

		ap_uint<TRANSACTIONWIDTH_OUT> *pPrediction = vecPrediction.at(sin).image_;

		ap_uint<TRANSACTIONWIDTH_IN> *pInputSinus = &(vecInputSinus.at(sin).at(0));

        TopLevel(pInputSinus ,pPrediction);

		std::cout << "Finished calculation for Sinus Nr." << sin + 1 << " of " << finish_sinus << std::endl;
	}
}



template
<
unsigned int TransactionWidth,
unsigned int ClassLabelWidth
>
void CalculateAccuracy(std::vector<InputImage<ap_uint<TransactionWidth>>>& vecPredictedImages, std::vector<InputImage<unsigned int>>& vecGTImages, std::vector<unsigned int>& errors, float& false_positive_rate, float& false_negative_rate)
{
	unsigned int start_sinus = 0;
	unsigned int finish_sinus;
	unsigned int false_negative = 0;
	unsigned int false_positive = 0;
	unsigned int total_negative = 0;
	unsigned int total_positive = 0;

	if(start_sinus == 0)
		finish_sinus = vecPredictedImages.size();
	else
		finish_sinus = start_sinus + 1;// selective execution of a single image

	for(unsigned int sin = start_sinus; sin < finish_sinus; sin++)
	{

		unsigned int gt = *vecGTImages.at(sin).image_;
		unsigned int guess = static_cast<unsigned int>(*vecPredictedImages.at(sin).image_);

		if(gt == guess){
			errors.push_back(0);
			if(gt == 0){
				total_negative++;
			}else{
				total_positive++;
			}
		}else{
			errors.push_back(1);
			if(gt == 0){
				total_negative++;
				false_positive++;
			}else{
				total_positive++;
				false_negative++;
			}
		}
	}

	if(total_negative != 0){
		false_negative_rate = ((float) false_negative) / ((float) total_negative);
	}else{
		false_negative_rate = 0;
	}


	if(total_positive != 0){
		false_positive_rate = ((float) false_positive) / ((float) total_positive);
	}else{
		false_positive_rate = 0;
	}
}
