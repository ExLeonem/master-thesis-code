
import numpy as np
import pytest
from modules.data.transformer import Pipeline, select_classes, image_channels, inputs_to_type


class TestClassSelectTransformer:

    def test_negative_class_selection(self):
        inputs = np.random.randn(10, 10)
        targets = np.random.randint(0, 2, 10)
        with pytest.raises(ValueError) as e:
            select_classes((inputs, targets), classes=-1)

    
    def test_select_more_classes_than_available(self):
        classes_to_select = 5
        num_classes_available = 2
        inputs = np.random.randn(10, 10)
        targets = np.random.randint(0, num_classes_available, 10)
        with pytest.raises(ValueError) as e:
            select_classes((inputs, targets), classes=classes_to_select)

    
    def test_select_zero_classes(self):
        classes_to_select = 0
        num_classes_available = 4
        inputs = np.random.randn(10, 10)
        targets = np.random.randint(0, num_classes_available, 10)
        with pytest.raises(ValueError) as e:
            select_classes((inputs, targets), classes=classes_to_select)

        
    def test_select_same_amount_as_available(self):
        num_classes_available = 3
        inputs = np.random.randn(10, 10)
        targets = np.random.randint(0, num_classes_available, 10)
        select_classes((inputs, targets), classes=num_classes_available)

    

class TestImageChannelsTransformation:
    """Test automatically adding grey-scale channel"""

    def test_invalid_dimension_space(self):
        inputs = np.random.randn(10, 28)
        targets = np.random.randn(10)
        with pytest.raises(ValueError) as e:
            image_channels((inputs, targets))


    def test_grey_scale_channel_added(self):
        # (batch, height, width)
        inputs = np.random.randn(10, 28, 28)
        targets = np.random.randn(10)
        (new_inputs, new_targets) = image_channels((inputs, targets))

        # Added the image channel dimension
        assert len(new_inputs.shape) == 4


    def test_transform_rgb_image(self):
        inputs = np.random.randn(10, 28, 28, 3)
        targets = np.random.randn(10)
        (new_inputs, new_targets) = image_channels((inputs, targets))

        assert inputs.shape == new_inputs.shape
        assert targets.shape == new_targets.shape


class TestInputToType:


    def test_int_to_float(self):
        inputs = np.random.randint(0, 10, (10, 10))
        targets = np.random.randint(0, 2, 10)

        (new_inputs, new_targets) = inputs_to_type((inputs, targets), dtype=np.float32)

        assert new_inputs.dtype != inputs.dtype
        assert new_inputs.dtype == np.float32


    def test_transform_to_already_existing(self):
        inputs = np.random.randint(0, 10, (10, 10))
        targets = np.random.randint(0, 2, 10)

        (new_inputs, new_targets) = inputs_to_type((inputs, targets), dtype=np.int64)
        assert new_inputs.dtype == inputs.dtype

    
    def test_transform_to_invalid_type(self):
        inputs = np.random.randint(0, 10, (10, 10))
        targets = np.random.randint(0, 2, 10)

        with pytest.raises(TypeError) as e:
            inputs_to_type((inputs, targets), dtype="hey")
        