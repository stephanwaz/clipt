_cl_plot_completion() {
    local cw="${COMP_WORDS[*]}"
    if [[ $cw != *">"* ]]; then
    if [[ ${COMP_WORDS[$COMP_CWORD]} != -* ]] && [[ ${COMP_WORDS[$COMP_CWORD-1]} == -* ]]; then
        :
    else
        local IFS=$'
'
        COMPREPLY=( $( env COMP_WORDS="${COMP_WORDS[*]}" \
                    COMP_CWORD=$COMP_CWORD \
                    _CL_PLOT_COMPLETE=complete $1 ) )
    fi
    fi
    return 0
}

_cl_plot_completionetup() {
    local COMPLETION_OPTIONS=""
    local BASH_VERSION_ARR=(${BASH_VERSION//./ })
    # Only BASH version 4.4 and later have the nosort option.
    if [ ${BASH_VERSION_ARR[0]} -gt 4 ] || ([ ${BASH_VERSION_ARR[0]} -eq 4 ] && [ ${BASH_VERSION_ARR[1]} -ge 4 ]); then
        COMPLETION_OPTIONS="-o nosort"
    fi

    complete $COMPLETION_OPTIONS -o default -F _cl_plot_completion cl_plot
}

_cl_plot_completionetup;
