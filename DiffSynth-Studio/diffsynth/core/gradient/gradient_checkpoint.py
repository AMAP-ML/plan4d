import torch


try:
    import deepspeed
    _HAS_DEEPSPEED = True
except ModuleNotFoundError:
    _HAS_DEEPSPEED = False


def create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)
    return custom_forward


def create_custom_forward_use_reentrant(module, kwargs=None):
    model_kwargs = {} if kwargs is None else kwargs

    def custom_forward(*inputs):
        return module(*inputs, **model_kwargs)
    return custom_forward


def create_custom_forward_for_deepspeed(module, args, kwargs, clone_deepspeed_tensor_kwargs=()):
    tensor_inputs = []
    positional_specs = []
    for value in args:
        if torch.is_tensor(value):
            positional_specs.append(("tensor", None))
            tensor_inputs.append(value)
        else:
            positional_specs.append(("value", value))

    clone_deepspeed_tensor_kwargs = set(clone_deepspeed_tensor_kwargs)
    keyword_specs = []
    for key, value in kwargs.items():
        if torch.is_tensor(value):
            keyword_specs.append((key, "tensor", None))
            tensor_inputs.append(value.clone() if key in clone_deepspeed_tensor_kwargs else value)
        else:
            keyword_specs.append((key, "value", value))

    def custom_forward(*tensor_values):
        tensor_iter = iter(tensor_values)
        rebuilt_args = []
        for kind, value in positional_specs:
            if kind == "tensor":
                rebuilt_args.append(next(tensor_iter))
            else:
                rebuilt_args.append(value)

        rebuilt_kwargs = {}
        for key, kind, value in keyword_specs:
            if kind == "tensor":
                rebuilt_kwargs[key] = next(tensor_iter)
            else:
                rebuilt_kwargs[key] = value

        return module(*rebuilt_args, **rebuilt_kwargs)

    return custom_forward, tuple(tensor_inputs)


def judge_args_requires_grad(*args):
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.requires_grad:
            return True
    return False


def collect_tensor_inputs(args, kwargs):
    tensor_inputs = []
    for value in args:
        if torch.is_tensor(value):
            tensor_inputs.append(value)
    for value in kwargs.values():
        if torch.is_tensor(value):
            tensor_inputs.append(value)
    return tuple(tensor_inputs)


def should_use_reentrant_checkpoint(module):
    has_parameter = False
    for parameter in module.parameters():
        has_parameter = True
        if not parameter.requires_grad:
            return True
    return False if has_parameter else False


def checkpoint_with_torch(
    model,
    args,
    model_kwargs,
    *,
    use_reentrant,
):
    if use_reentrant:
        return torch.utils.checkpoint.checkpoint(
            create_custom_forward_use_reentrant(model, model_kwargs),
            *args,
            use_reentrant=True,
        )
    return torch.utils.checkpoint.checkpoint(
        create_custom_forward(model),
        *args,
        **model_kwargs,
        use_reentrant=False,
    )


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    *args,
    **kwargs,
):
    model_kwargs = dict(kwargs)
    clone_deepspeed_tensor_kwargs = tuple(model_kwargs.pop("clone_deepspeed_tensor_kwargs", ()))
    use_reentrant = should_use_reentrant_checkpoint(model)
    tensor_inputs = collect_tensor_inputs(args, model_kwargs)

    if use_gradient_checkpointing and _HAS_DEEPSPEED and deepspeed.checkpointing.is_configured():
        custom_forward, tensor_inputs = create_custom_forward_for_deepspeed(
            model,
            args,
            model_kwargs,
            clone_deepspeed_tensor_kwargs=clone_deepspeed_tensor_kwargs,
        )
        if len(tensor_inputs) == 0 or not judge_args_requires_grad(*tensor_inputs):
            # get the first grad_enabled tensor from un_checkpointed forward
            model_output = model(*args, **model_kwargs)
        else:
            model_output = deepspeed.checkpointing.checkpoint(
                custom_forward,
                *tensor_inputs,
            )
        return model_output
    if (use_gradient_checkpointing or use_gradient_checkpointing_offload) and (
        len(tensor_inputs) == 0 or not judge_args_requires_grad(*tensor_inputs)
    ):
        return model(*args, **model_kwargs)
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            model_output = checkpoint_with_torch(
                model,
                args,
                model_kwargs,
                use_reentrant=use_reentrant,
            )
    elif use_gradient_checkpointing:
        model_output = checkpoint_with_torch(
            model,
            args,
            model_kwargs,
            use_reentrant=use_reentrant,
        )
    else:
        model_output = model(*args, **model_kwargs)
    return model_output
