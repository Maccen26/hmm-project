import jax
jax.config.update("jax_enable_x64", True)

from src.api.v4.utils import load_y_data
from src.api.v4 import HMM, GaussEmission, StaticTransitionHigherOrder
from src.api.v4.transitions.static_transition_higher_order import decode_possible_transitions

import jax.numpy as jnp
from drivers.utils import  load_model, save_model

def run_second_order_hmm(model_name: str):
    print(f"Starting {model_name} model run...")
    ys = load_y_data()

    #Initiing parameters 
    PRE_MODEL_PATH = "results/models/ar_2_hmm.pkl"

    model = load_model(PRE_MODEL_PATH)

    params = model.params 
    transition_logits = list(params.transition.transition_logits)


    #raise NotImplementedError("This model is not fully implemented yet. The phi values are not being updated during fitting. This is a known issue and will be fixed in the future.")
    ## Initiating the HMM model
    transition = StaticTransitionHigherOrder(jnp.asarray(transition_logits * 4), order=2)
    emission = GaussEmission.from_params(mu=jnp.sort(jnp.repeat(params.emission.mu(0, ys), 4)), sigma=jnp.sort(jnp.repeat(params.emission.sigma(0, ys), 4)))
    model = HMM(transition=transition, emission=emission) 

    ## Fitting the model
    frozen_params = {
    "mu0": False
    }

    print(f"Fitting {model_name} model...")
    model.fit(ys =ys, frozen=frozen_params)

    print(f"Finished fitting {model_name} model! The following parameters were found")
    print("Transition matrix:")
    transition_matrix = model.transition.transition_matrix(0, ys) 
    state_transions = decode_possible_transitions(transition_matrix, order = 2 )
    for state, transition_list in state_transions.items():
        print(f"From {state}:")
        for to_state, prob in transition_list:
            print(f"  To {to_state}: {prob:.2f}") 
        print("----------")

    print("Emission means:")
    print(model.emission.mu(0, ys))
    print("Emission stds:")
    print(model.emission.sigma(0, ys))
    print("------------------------------------")

    return model


if __name__ == "__main__":
    model_name = "second_order_hmm"
    PATH = f"results/models/{model_name}.pkl"
    model = run_second_order_hmm(model_name)
    save_model(model, PATH)