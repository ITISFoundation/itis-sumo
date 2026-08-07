### Useful functions to couple Python and Dakota - to use accross different scripts & notebooks
import logging
import shutil
from pathlib import Path
from typing import Literal

_logger = logging.getLogger(__name__)


def start_dakota_file(
    top_method_pointer: str | None = None,
    results_file_name: str | None = None,
) -> str:
    """Make the start of a Dakota input file - making it produce a 'results.dat' file"""
    if results_file_name is None:
        results_file_name = "results.dat"

    return f"""
    environment
        tabular_data
            tabular_data_file = '{results_file_name}'
        {f"top_method_pointer = '{top_method_pointer}'" if top_method_pointer is not None else ""}
    """


def add_adaptive_sampling(
    N_ADAPT: int,
    training_samples_file: str | None = None,
    id_method: str = "ADAPTIVE_SAMPLING",
    model_pointer: str = "TRUE_MODEL",
    seed=43,
) -> str:
    s = f"""

    method
        id_method = '{id_method}'
        adaptive_sampling
            max_iterations {N_ADAPT}
            samples_on_emulator = 1000
            fitness_metric predicted_variance
    """

    if training_samples_file:
        s += f"""
            import_build_points_file
            "{training_samples_file}"
            {"custom_annotated header use_variable_labels eval_id" if "_processed.txt" in training_samples_file else ""}  # only if processed file
            """
    else:
        raise ValueError("Training samples file must be provided for adaptive sampling")

    s += f"""
            model_pointer = "{model_pointer}"
            seed {seed}

            export_approx_points_file "predictions.dat"

            # response_levels # dont know how to use this
            # probability_levels # dont know how to use this
            # gen_reliability_levels # dont know how to use this

    """
    return s


def add_continuous_variables(
    variables: list[str],
    id_variables="VARIABLES",
    initial_points: list[float] | None = None,
    lower_bounds: list[float] | None = None,
    upper_bounds: list[float] | None = None,
    type: Literal["design", "state"] = "design",
) -> str:
    vars_str = f"""

        variables
            continuous_{type} = {len(variables)}
            id_variables = '{id_variables}'
                descriptors {" ".join([f"'{v}'" for v in variables])}"""
    if initial_points is not None:
        assert len(initial_points) == len(variables)
        vars_str += f"""
                initial_point     {" ".join([str(ip) for ip in initial_points])}"""
    if lower_bounds is not None:
        assert len(lower_bounds) == len(variables)
        vars_str += f"""
                lower_bounds      {" ".join([str(lb) for lb in lower_bounds])}"""
    if upper_bounds is not None:
        assert len(upper_bounds) == len(variables)
        vars_str += f"""
                upper_bounds      {" ".join([str(ub) for ub in upper_bounds])}"""
    return vars_str


def add_responses(descriptors: list[str]) -> str:
    descriptors = descriptors if isinstance(descriptors, list) else [descriptors]
    return f"""

        responses
            id_responses = 'RESPONSES'
            descriptors {" ".join([f"'{d}'" for d in descriptors])}
            objective_functions = {len(descriptors)}
            no_gradients
            no_hessians
        """


def add_surrogate_model(
    id_model: str = "SURR_MODEL",
    surrogate_type: str = "gaussian_process surfpack",
    sumo_import_name: str | None = None,
    sumo_export_name: str | None = None,
    export_import_format: str = "text_archive",
    training_samples_file: str | None = None,
    id_sampling_method: str | None = None,
    cross_validation_folds: int | None = None,
) -> str:
    conf = f"""
        model
            id_model '{id_model}'
            surrogate global
                {surrogate_type}
        """

    if sumo_import_name:
        conf += f"""
                import_model {export_import_format} filename_prefix='{sumo_import_name}'
        """
        # When importing a surrogate model, it is crucial that the global surrogate model part
        # of the Dakota input file be identical for export and import,
        # except for changing export_model and its child keywords to those needed for import_model.
        # Any other keywords such as specifying a dace_iterator or imported points
        # must remain intact to satisfy internal surrogate constructor requirements.
    else:
        if sumo_export_name:
            conf += f"""
                    export_model filename_prefix='{sumo_export_name}' formats={export_import_format}
            """

    if cross_validation_folds:
        conf += f"""
                cross_validation folds = {cross_validation_folds}
                metrics = "root_mean_squared" "sum_abs" "mean_abs" "max_abs" "rsquared"
        """

    if training_samples_file is None:
        _logger.info(
            "No training samples file provided, using sampling to build the surrogate instead"
        )
        assert id_sampling_method is not None, (
            "id_sampling must be provided if no training samples file is provided"
        )
        conf += f"""
                dace_method_pointer = '{id_sampling_method}'"
                """
    else:
        conf += f"""
                import_build_points_file
                    '{training_samples_file}'
                    custom_annotated header use_variable_labels {"eval_id" if "processed" not in training_samples_file else ""}"""

        ### DONT KNOW HOW TO USE THIS YET
        # if id_truth_model is None:
        #     print(
        #         "No truth model to evaluate samples provided for the surrogate model "
        #         "- it must then be provided within the sampling method"
        #     )
        #     assert (
        #         id_sampling_method is not None
        #     ), "id_sampling must be provided if no truth model is provided"

        #             {f"truth_model_pointer =  '{id_truth_model}' "
        #             if id_truth_model is not None else
        #             f"dace_method_pointer = '{id_sampling_method}'"}

    conf += f"""
                export_approx_points_file "predictions.dat"
                {'export_approx_variance_file "variances.dat"' if "gaussian_process" in surrogate_type else ""}
        """

    return conf


def add_sampling_method(
    id_method: str = "SAMPLING",
    sampling_method: str = "lhs",
    model_pointer: str | None = None,
    num_samples: int = 10,
    seed: int = 1234,
    refinement: bool = False,
    refinement_samples: int | None = None,
    variance_based_decomp: bool = False,
) -> str:
    conf = f"""
        method
            id_method = '{id_method}'
            sample_type
                {sampling_method}
            sampling
                samples = {num_samples}
                {f"seed = {seed}" if seed is not None else ""}
                {"variance_based_decomp" if variance_based_decomp else ""}
            {f'model_pointer = "{model_pointer}"' if model_pointer is not None else ""}
        """

    if variance_based_decomp:
        assert sampling_method == "lhs", "variance_based_decomp only available for LHS sampling"

    if refinement:
        assert sampling_method == "lhs", "Refinement only available for LHS"
        assert refinement_samples == num_samples, (
            "Refinement samples must be equal to num_samples. Multiple-step refinement not yet implemented"
        )
        conf += f"""
            refinement_samples = {refinement_samples}
        """
        _logger.warning(
            "WARNING: to use refinement, make sure you provide reset files as input for Dakota"
        )

    return conf


def add_evaluation_method(
    input_file: str,
    model_pointer: str = "SURR_MODEL",
    includes_eval_id: bool = False,
) -> str:
    eval_str = f"""
        method
            id_method "EVALUATION"
            model_pointer '{model_pointer}'
        """
    if input_file is not None:
        eval_str += f"""
            list_parameter_study
                import_points_file
                    ## this file should be wo responses!!
                    '{input_file}'
                    custom_annotated header {"eval_id" if includes_eval_id else ""}
        """
    return eval_str


def add_moga_method(
    populationSize=32,
    maxIterations=100,
    max_function_evaluations=None,
    fitnessType: Literal["layer_rank", "domination_count"] = "layer_rank",
    replacementType: Literal[
        "elitist", "unique_roulette_wheel", "below_limit"
    ] = "elitist",  # N.B. roulette_wheel not working in dakota 6.19
    id_method="MOGA",
    seed=12345,
    ############ not exposed in MMUX ####################
    max_designs=32,
    crossover_type: str = "multi_point_real 5",
    mutation_type: str = "offset_uniform",
    #####################################################
):
    if max_function_evaluations:
        _logger.warning("Max Function Evaluations for MOGA is deprecated; will be ignored")
    return f"""
        method
            id_method = '{id_method}'
            moga
            population_size = {populationSize} # Set the initial population size in JEGA methods
            max_iterations = {maxIterations}

            ## hyperparameters taken from Medtronic's pulse shape optimization
            fitness_type
                {fitnessType}
            crossover_type
                {crossover_type}
            mutation_type
                {mutation_type}
            niching_type
                max_designs {max_designs} # Limit number of solutions to remain in the population
            replacement_type
                {replacementType} {6 if replacementType == "below_limit" else ""} # Only used with below_limit replacement_type
            seed = {seed}
        """


def add_evaluator_model(
    id_model="TRUE_MODEL",
    interface_pointer="INTERFACE",
    variables_pointer="VARIABLES",
    responses_pointer="RESPONSES",
):
    return f"""
        model
            id_model = '{id_model}'
            single
                interface_pointer = '{interface_pointer}'
                variables_pointer = '{variables_pointer}'
                responses_pointer = '{responses_pointer}'
        """


def write_to_file(dakota_conf_text, dakota_conf_path):
    dakota_conf_path = Path(dakota_conf_path)
    dakota_conf_path.write_text(dakota_conf_text)


############## COMMON WORKFLOWS ######################
def create_sumo_evaluation_conffile(
    build_file: Path,
    # surrogate_type: Optional[str] = None, ## for now, always GP
    # TODO be able to load sumo (instead of building every time)
    samples_file: Path,
    input_variables: list[str],
    output_responses: list[str],
    dakota_conf_file: str | Path | None = None,
    sumo_import_name: str | None = None,
    sumo_export_name: str | None = None,
) -> str:
    dakota_conf = start_dakota_file()
    dakota_conf += add_surrogate_model(
        training_samples_file=str(build_file.resolve()),
        sumo_export_name=sumo_export_name,
        sumo_import_name=sumo_import_name,
    )
    dakota_conf += add_evaluation_method(str(samples_file.resolve()))
    dakota_conf += add_continuous_variables(variables=input_variables)
    dakota_conf += add_responses(output_responses)
    # dakota_conf += add_python_interface() ## no need to run anything outside dakota!

    if dakota_conf_file:
        write_to_file(dakota_conf, dakota_conf_file)
    return dakota_conf


def create_uq_propagation_conffile(
    build_file: Path,
    # surrogate_type: Optional[str] = None, ## for now, always GP
    # TODO be able to load sumo (instead of building every time)
    input_variables: list[str],
    input_means: dict[str, float],
    input_stds: dict[str, float],
    output_responses: list[str],
    n_samples: int = 10000,
    dakota_conf_file: str | Path | None = None,
) -> str:
    dakota_conf = start_dakota_file()
    dakota_conf += add_surrogate_model(training_samples_file=str(build_file.resolve()))
    dakota_conf += add_sampling_method(num_samples=n_samples)
    ## TODO this is only NORMAL uncertain -- need to generalize if we want to include other types of input distributions
    dakota_conf += f"""
        variables
            id_variables = "VARIABLES"
            active uncertain
            normal_uncertain = {len(input_variables)}
                descriptors {" ".join([f"'{var}'" for var in input_variables])}
                means {" ".join([str(input_means[var]) for var in input_variables])}
                std_deviations {" ".join([str(input_stds[var]) for var in input_variables])}
        """
    dakota_conf += add_responses(output_responses)

    if dakota_conf_file:
        write_to_file(dakota_conf, dakota_conf_file)

    return dakota_conf


def create_sumo_crossvalidation_conffile(
    build_file: Path,
    input_variables: list[str],
    output_responses: list[str],
    dakota_conf_file: str | Path | None = None,
    N_CROSS_VALIDATION=5,
):
    dakota_conf = start_dakota_file()
    dakota_conf += add_surrogate_model(
        training_samples_file=str(build_file.resolve()),
        cross_validation_folds=N_CROSS_VALIDATION,
    )
    from itis_sumo.data.funs_data_processing import process_input_file

    JUST_INPUTS_FILE = process_input_file(
        build_file,
        columns_to_remove=output_responses,
    )
    dakota_conf += add_evaluation_method(
        str(JUST_INPUTS_FILE.resolve()),
        includes_eval_id=False,  ## just to have some method, otherwise Dakota gives error
    )
    dakota_conf += add_continuous_variables(
        variables=input_variables,
    )
    dakota_conf += add_responses(output_responses)

    if dakota_conf_file:
        write_to_file(dakota_conf, dakota_conf_file)

    return dakota_conf


def create_sumo_manual_crossvalidation_conffile(
    fold_run_dir: Path,
    build_file: Path,
    input_variables: list[str],
    output_response: str,
    validation_indices: list[int],
    dakota_conf_file: str | Path | None = None,
):
    from itis_sumo.data.funs_data_processing import load_data, process_input_file

    dakota_conf = start_dakota_file()
    n_samples = len(load_data(build_file))
    _logger.debug("Number of samples in the build file: %d", n_samples)
    ## filter OUT the validation indices from the build_file
    TRAINING_SAMPLES_FILE = process_input_file(
        build_file,
        keep_idxs=[i for i in range(n_samples) if i not in validation_indices],
        columns_to_keep=input_variables + [output_response],
        suffix="training",
    )
    TRAINING_SAMPLES_FILE = Path(
        shutil.move(
            str(TRAINING_SAMPLES_FILE.resolve()),
            str(fold_run_dir / TRAINING_SAMPLES_FILE.name),  # move to the fold run dir
        )
    )
    dakota_conf += add_surrogate_model(
        training_samples_file=str(TRAINING_SAMPLES_FILE.resolve()),
    )
    JUST_INPUTS_FILE = process_input_file(
        build_file,
        keep_idxs=validation_indices,
        columns_to_keep=input_variables,
        suffix="validation",
    )
    JUST_INPUTS_FILE = Path(
        shutil.move(
            str(JUST_INPUTS_FILE.resolve()),
            str(fold_run_dir / JUST_INPUTS_FILE.name),  # move to the fold run dir
        )
    )
    ## For some freaking reason, there are 6 points in JUST_INPUTS_FILE and 9 get evaluated!!
    _logger.debug("Build file: %s", build_file)
    _logger.debug("Training samples file: %s", TRAINING_SAMPLES_FILE)
    _logger.debug("Just inputs file: %s", JUST_INPUTS_FILE)
    dakota_conf += add_evaluation_method(
        str(JUST_INPUTS_FILE.resolve()),
        includes_eval_id=False,
    )
    dakota_conf += add_continuous_variables(
        variables=input_variables,
    )
    dakota_conf += add_responses([output_response])

    if dakota_conf_file:
        write_to_file(dakota_conf, dakota_conf_file)

    return dakota_conf


def create_moga_optimization_conffile(
    build_file: Path,
    input_variables: list[str],
    output_responses: list[str],
    moga_kwargs: dict,
    lower_bounds: list | None = None,
    upper_bounds: list | None = None,
    dakota_conf_file: str | Path | None = None,
):
    dakota_conf = start_dakota_file()
    dakota_conf += add_surrogate_model(training_samples_file=str(build_file.resolve()))
    dakota_conf += add_moga_method(**moga_kwargs)
    dakota_conf += add_continuous_variables(
        variables=input_variables,
        lower_bounds=(lower_bounds if lower_bounds else [0.0 for _ in range(len(input_variables))]),
        upper_bounds=(upper_bounds if upper_bounds else [1.0 for _ in range(len(input_variables))]),
    )
    dakota_conf += add_responses(descriptors=output_responses)
    # dakota_conf += add_python_interface("model", batch_mode=batch_mode)

    if dakota_conf_file:
        write_to_file(dakota_conf, dakota_conf_file)
    return dakota_conf

