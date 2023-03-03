# pylint: disable = (missing-module-docstring)

import torch
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    GPTNeoForCausalLM,
    T5ForConditionalGeneration,
)


class CustomFCN(torch.nn.Module):
    """Custom fully connected network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.flatten = torch.nn.Flatten()
        self.layer_1 = torch.nn.Linear(input_size, hidden_size)
        self.relu_1 = torch.nn.ReLU()
        self.dropout_1 = torch.nn.Dropout(0.2)
        self.layer_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu_2 = torch.nn.ReLU()
        self.dropout_2 = torch.nn.Dropout(0.2)
        self.layer_3 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu_3 = torch.nn.ReLU()
        self.dropout_3 = torch.nn.Dropout(0.2)
        self.layer_4 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Forward pass of custom FCN network.

        Args:
            sample: Sample to process.

        Returns:
            Tensor result.
        """
        sample = self.flatten(sample)
        sample = self.layer_1(sample)
        sample = self.relu_1(sample)
        sample = self.dropout_1(sample)
        sample = self.layer_2(sample)
        sample = self.relu_2(sample)
        sample = self.dropout_2(sample)
        sample = self.layer_3(sample)
        sample = self.relu_3(sample)
        sample = self.dropout_3(sample)
        out = self.layer_4(sample)
        # no activation and no softmax at the end
        return out


class CustomCNN(torch.nn.Module):
    """Custom CNN network."""

    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 12, 5)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(12, 24, 5)
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        self.conv4 = torch.nn.Conv2d(24, 1, 5)
        self.fc1 = torch.nn.Linear(400, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, num_classes)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Forward pass of custom CNN network.

        Args:
            sample: Sample to process.

        Returns:
            Tensor result.
        """
        sample = self.pool1(torch.nn.functional.relu(self.conv1(sample)))
        sample = self.pool2(torch.nn.functional.relu(self.conv2(sample)))
        sample = self.pool3(torch.nn.functional.relu(self.conv3(sample)))
        sample = torch.nn.functional.relu(self.conv4(sample))
        # flatten all dimensions except batch
        sample = torch.flatten(sample, 1)  # pylint: disable = (no-member)
        sample = torch.nn.functional.relu(self.fc1(sample))
        sample = torch.nn.functional.relu(self.fc2(sample))
        sample = self.fc3(sample)
        return sample


class CustomLSTM(torch.nn.Module):
    """Custom LSTM network."""

    # https://www.kaggle.com/code/andradaolteanu/pytorch-rnns-and-lstms-explained-acc-0-99
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layer_size: int,
        num_classes: int,
        device: torch.device,  # pylint: disable = (no-member)
        batch_size: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.output_size = num_classes
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.device = device

        self.lstm = torch.nn.LSTM(
            input_size * input_size,
            hidden_size,
            layer_size,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        # Create FNN
        if self.bidirectional:
            self.fnn = torch.nn.Linear(hidden_size * 2, num_classes)
        else:
            self.fnn = torch.nn.Linear(hidden_size, num_classes)

        if self.bidirectional:
            self.hidden_state = torch.zeros(  # pylint: disable = (no-member)
                self.layer_size * 2,
                self.batch_size,
                self.hidden_size,
            ).to(self.device)
            self.cell_state = torch.zeros(  # pylint: disable = (no-member)
                self.layer_size * 2,
                self.batch_size,
                self.hidden_size,
            ).to(self.device)
        else:
            self.hidden_state = torch.zeros(  # pylint: disable = (no-member)
                self.layer_size,
                self.batch_size,
                self.hidden_size,
            ).to(self.device)
            self.cell_state = torch.zeros(  # pylint: disable = (no-member)
                self.layer_size,
                self.batch_size,
                self.hidden_size,
            ).to(self.device)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Forward pass of custom LSTM network.

        Args:
            sample: Sample to process.

        Returns:
            Tensor result.
        """
        sample = sample.reshape(sample.shape[0], sample.shape[1], -1)
        output, _ = self.lstm(sample, (self.hidden_state, self.cell_state))

        # FNN
        output = self.fnn(output[:, -1, :])
        return output


class Bert(torch.nn.Module):
    """BERT wrapper."""

    def __init__(self, model_name: str = "textattack/bert-base-uncased-imdb"):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, torchscript=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of BERT model.

        Args:
            input_ids: Indices of input sequence tokens in the vocabulary.
            token_type_ids: Segment token indices to indicate first and second
                portions of the inputs. Indices are selected in [0, 1]
            attention_mask: Mask to avoid performing attention on padding token
                indices. Mask values selected in [0, 1].

        Returns:
            Tensor result.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]


class T5(torch.nn.Module):
    """T5 wrapper."""

    def __init__(
        self,
        model_name: str = "t5-base",
        max_length: int = 200,
        min_length: int = 100,
        num_beams: int = 4,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, torchscript=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,  # pylint: disable = (unused-argument)
    ) -> torch.Tensor:
        """Forward pass of T5 model.

        Args:
            input_ids: Indices of input sequence tokens in the vocabulary.
            token_type_ids: Segment token indices to indicate first and second
                portions of the inputs. Indices are selected in [0, 1]

        Returns:
            Tensor result.
        """
        outputs = self.model.generate(
            input_ids,
            max_length=self.max_length,
            min_length=self.min_length,
            num_beams=self.num_beams,
        )
        return outputs[0]


class GPTNeo(torch.nn.Module):
    """GPTNeo wrapper."""

    def __init__(
        self,
        model_name: str = "EleutherAI/gpt-neo-125M",
        max_length: int = 200,
        min_length: int = 100,
        num_beams: int = 4,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams
        self.model = GPTNeoForCausalLM.from_pretrained(
            self.model_name, torchscript=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of GPTNeo model.

        Args:
            input_ids: Indices of input sequence tokens in the vocabulary.
            token_type_ids: Segment token indices to indicate first and second
                portions of the inputs. Indices are selected in [0, 1]

        Returns:
            Tensor result.
        """
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            min_length=self.min_length,
            num_beams=self.num_beams,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return outputs[0]
