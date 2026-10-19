"""PyTorch network, checkpoint and scientific loss regression tests."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from xirt.training import make_optimizers, task_loss
from xirt.xirtnet import params_to_df, read_checkpoint, xiRTNET


@pytest.fixture
def params():
    """Return a small legacy-shaped multi-task architecture."""
    with open(Path(__file__).parent / "fixtures/xirt_params_3RT.yaml") as stream:
        result = yaml.safe_load(stream)
    result["learning"]["device"] = "cpu"
    return result


def network(params, siamese=True, alphabet=25):
    """Build a tiny network using the public compatibility API."""
    result = xiRTNET(params, input_dim=6, embedding_dim=alphabet)
    result.build_model(siamese=siamese, alphabet=alphabet)
    result.compile()
    return result


@pytest.mark.parametrize("container", [list, tuple])
def test_siamese_branch_containers(params, container):
    params["predictions"] = {"fractions": [], "continues": ["rp"]}
    model = network(params).model
    first = np.array([[0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]])
    second = first[::-1].copy()
    inputs = container([first, second])
    targets = np.ones((2, 1), dtype=np.float32)

    np.testing.assert_allclose(model.predict(inputs), model.predict((first, second)))
    np.testing.assert_allclose(
        model.evaluate(inputs, targets), model.evaluate((first, second), targets)
    )
    history = model.fit(
        inputs, targets, validation_data=(inputs, targets), epochs=1, batch_size=2, verbose=0
    ).history
    assert len(history["loss"]) == len(history["val_loss"]) == 1


def test_single_branch_row_lists(params):
    params["predictions"] = {"fractions": [], "continues": ["rp"]}
    model = network(params, siamese=False).model
    array = np.array([[0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]])
    rows = array.tolist()
    targets = np.ones((2, 1), dtype=np.float32)

    np.testing.assert_allclose(model.predict(rows), model.predict(array))
    np.testing.assert_allclose(model.predict((array,)), model.predict(array))
    np.testing.assert_allclose(model.evaluate(rows, targets), model.evaluate(array, targets))
    history = model.fit(
        rows, targets, validation_data=(rows, targets), epochs=1, batch_size=2, verbose=0
    ).history
    assert len(history["loss"]) == len(history["val_loss"]) == 1


@pytest.mark.parametrize("merge", ["add", "multiply", "average", "concatenate", "maximum"])
@pytest.mark.parametrize("rnn", ["GRU", "LSTM"])
def test_architecture(params, merge, rnn):
    params["siamese"]["merge_type"] = merge
    params["LSTM"]["type"] = rnn
    params["LSTM"]["nlayers"] = 3
    model = network(params).model
    inputs = np.array([[0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]])
    outputs = model.predict((inputs, inputs), batch_size=1)
    assert [v.shape for v in outputs] == [(2, 10), (2, 9), (2, 1)]
    assert np.isfinite(np.concatenate(outputs, axis=1)).all()
    if merge != "concatenate":
        swapped = model.predict((inputs[::-1].copy(), inputs))
        original = model.predict((inputs, inputs[::-1].copy()))
        for first, second in zip(original, swapped):
            np.testing.assert_allclose(first, second, atol=1e-6)


def test_padding_does_not_change_representation(params):
    model = network(params).model.eval()
    left = torch.tensor([[0, 0, 0, 1, 2, 3]])
    right = torch.tensor([[1, 2, 3, 0, 0, 0]])
    torch.testing.assert_close(model.encode(left), model.encode(right))
    assert torch.equal(model.encode(torch.zeros_like(left)), torch.zeros(1, 10))


def test_activity_regularization_is_included_in_evaluation_objective(params):
    params["predictions"] = {"continues": ["rp"], "fractions": []}
    params["LSTM"].update(lstm_bn=False, kernelregularizer_value=0.0, activityregularizer_value=0.5)
    params["dense"]["dense_bn"] = [False] * params["dense"]["nlayers"]
    params["dense"]["dropout"] = [0.0] * params["dense"]["nlayers"]
    params["dense"]["regularizer_value"] = [0.0] * params["dense"]["nlayers"]
    model = network(params).model
    inputs = np.array([[0, 0, 1, 2, 3, 4], [0, 0, 2, 3, 4, 5]])
    labels = (torch.ones(2, 1),)
    model.train()
    logits = model(torch.as_tensor(inputs), torch.as_tensor(inputs))
    training_loss, _ = model._losses(logits, labels)
    task = task_loss(logits["rp"], labels[0], "mse")
    expected_penalty = 0.5 * sum(value.square().sum() / len(value) for value in model.activity)
    assert expected_penalty.item() > 0
    assert training_loss.item() == pytest.approx(task.item() + expected_penalty.item())
    evaluation = model.evaluate((inputs, inputs), labels[0].numpy(), batch_size=2)
    assert evaluation[0] == pytest.approx(training_loss.item())
    assert model.training


def test_training_checkpoint_and_singleton(params, tmp_path):
    params["callbacks"]["callback_path"] = str(tmp_path)
    net = network(params)
    rng = np.random.default_rng(42)
    x = (rng.integers(1, 25, (9, 6)), rng.integers(1, 25, (9, 6)))
    y = [np.zeros((9, 10)), np.zeros((9, 9)), np.ones(9)]
    before = copy.deepcopy(net.model.state_dict())
    history = net.model.fit(
        x,
        y,
        validation_data=(x, y),
        epochs=2,
        batch_size=4,
        callbacks=net.get_callbacks("test"),
        verbose=0,
    ).history
    assert len(history["val_loss"]) == 2
    assert any(not torch.equal(value, net.model.state_dict()[key]) for key, value in before.items())
    restored = xiRTNET(params, 6, 25)
    restored.load_model(tmp_path / "xirt_model_test.pt")
    net.model.load_weights(tmp_path / "xirt_weights_test.pt")
    for first, second in zip(net.model.predict(x), restored.model.predict(x)):
        np.testing.assert_allclose(first, second)
    assert len(net.model.evaluate(x, y)) == len(net.model.metrics_names)
    assert read_checkpoint(tmp_path / "xirt_model_test.pt")["format_version"] == 2
    assert (tmp_path / "xirt_epochlog_test.log").exists()


@pytest.mark.parametrize(
    "name,cls",
    [
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("sgd", torch.optim.SGD),
        ("rmsprob", torch.optim.RMSprop),
        ("nadam", torch.optim.NAdam),
    ],
)
def test_optimizer(params, name, cls):
    params["learning"]["optimizer"] = name
    assert isinstance(network(params).model.optimizer, cls)


def test_muon_parameter_partition(params):
    model = network(params).model
    opts = make_optimizers(model, {"optimizer": "muon", "learningrate": 0.001})
    ids = [id(p) for opt in opts for group in opt.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    muon_ids = {id(p) for group in opts[0].param_groups for p in group["params"]}
    assert id(model.embedding.weight) not in muon_ids
    assert all(id(p) not in muon_ids for p in model.outputs.parameters())
    model.optimizers = opts
    x = np.ones((3, 6), dtype=int)
    model.fit((x, x), [np.zeros((3, 10)), np.zeros((3, 9)), np.ones(3)], batch_size=2, verbose=0)


@pytest.mark.parametrize(
    "loss,focal",
    [
        ("binary_crossentropy", "binary_focal"),
        ("categorical_crossentropy", "categorical_focal"),
    ],
)
def test_focal_limit_and_extreme_logits(loss, focal):
    logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    torch.testing.assert_close(
        task_loss(logits, targets, loss), task_loss(logits, targets, focal, gamma=0)
    )
    value = task_loss(logits, targets, focal)
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(logits.grad).all()


def test_adjust_preserves_encoder(params):
    net = network(params)
    old = net.model.count_params()
    embedding = net.model.embedding.weight.detach().clone()
    net.output_p["scx-dimension"] = 5
    net.output_p["hsax-dimension"] = 5
    net.adjust_model()
    assert old - net.model.count_params() == 99
    torch.testing.assert_close(embedding, net.model.embedding.weight)


def test_errors_are_explicit(params, tmp_path):
    model = network(params).model
    with pytest.raises(ValueError, match="integer indices"):
        model.predict((np.ones((2, 6)) * 0.5, np.ones((2, 6))))
    with pytest.raises(ValueError, match="matching sample"):
        model.predict((np.ones((2, 6)), np.ones((3, 6))))
    with pytest.raises(ValueError, match="HDF5"):
        model.load_weights("legacy.h5")
    with pytest.raises(ValueError, match="empty"):
        model.fit((np.empty((0, 6)), np.empty((0, 6))), [])
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        make_optimizers(model, {"optimizer": "unknown", "learningrate": 0.001})
    params["dense"]["dropout"] = []
    with pytest.raises(ValueError, match="must contain"):
        xiRTNET(params, 6, 25)


def test_embedding_route_and_empty_predictions(params):
    params["encoder"] = {"route": "esm-c", "embedding_dim": 4}
    model = network(params).model
    arrays = model.predict((np.empty((0, 4)), np.empty((0, 4))))
    assert [a.shape for a in arrays] == [(0, 10), (0, 9), (0, 1)]
    with pytest.raises(ValueError, match="shape"):
        model.predict((np.ones((2, 3)), np.ones((2, 4))))


def test_params_to_df(tmp_path):
    table = params_to_df(
        Path(__file__).parent / "fixtures/xirt_params_3RT.yaml", tmp_path / "params.csv"
    )
    assert not table.empty
    assert (tmp_path / "params.csv").exists()
