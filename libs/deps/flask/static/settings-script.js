function prevent_submit_and_unfocus(e) {
    document.activeElement.blur()
    e.preventDefault();
}

function isNumeric(str) {
    if (typeof str != "string") return false; // we only process strings!
    return (
        (!isNaN(str) && // use type coercion to parse the _entirety_ of the string (`parseFloat` alone does not do this)...
            !isNaN(parseFloat(str))) || // ...and ensure strings of whitespace fail
        str === ""
    ); // except completly empty to ease the retyping of the first digit
}

$(window).on("load", () => {
    const mspr_input = document.getElementById("motor-steps-per-rotation-input");
    const dpr_input = document.getElementById("distance-per-rotation-input");
    let mspr_value = mspr_input.value;
    let dpr_value = dpr_input.value;

    // event handlers

    $("form").submit(prevent_submit_and_unfocus);
    
    $(".submit-button").on("click", function () {
        const attr_id =  $(this).attr("for")
        
        const id_name_table = {
            "gpio-default-on-input": "GPIO-default-on",
            "gpio-motor-pins-input": "GPIO-motor-pins",
            "gpio-camera-pin-input": "GPIO-camera-pin"
        }

        const attr_name = id_name_table[attr_id];

        let value = $(`#${attr_id}`).val();
        if (attr_id === "gpio-motor-pins-input") {
            value = `[${value}]`
        }

        $.post(`/settings/${attr_name}/${value}`);
    })

    $("#units-input").on("change", function () {
        
        $.post(`/settings/motor-rotation-units/${Math.log10(this.value)}`);
        $(".unit").html($("#units-input option:selected").text());
    });

    
    $(mspr_input).on("keyup", function (e) {
        if (isNumeric(mspr_input.value) === false) {
            mspr_input.value = mspr_value;
        } else {
            if (mspr_input.value === "") {
            } else {
                mspr_value = parseFloat(mspr_input.value);
            }
        }

        $.post(`/settings/steps-per-motor-rotation/${mspr_input.value}`);
    });

    $(dpr_input).on("keyup", function (e) {
        if (isNumeric(dpr_input.value) === false) {
            dpr_input.value = dpr_value;
        } else {
            if (dpr_input.value === "") {
            } else {
                dpr_value = parseFloat(dpr_input.value);
            }
        }

        $.post(`/settings/distance-per-motor-rotation/${dpr_input.value}`);
    });

    // load saved values

    $.get("/settings/GPIO-default-on", (async=false), (value) => {
        $("#gpio-default-on-input").val(value);
    });

    $.get("/settings/GPIO-motor-pins", (async=false), (value) => {
        $("#gpio-motor-pins-input").val(value.substring(1, value.length-1));
    });

    $.get("/settings/GPIO-camera-pin", (async=false), (value) => {
        $("#gpio-camera-pin-input").val(parseInt(value))
    });

    $.get("/settings/motor-rotation-units", (async = false), (value) => {
        $("#units-input").val(Math.pow(10, parseInt(value)), value)
        $(".unit").html($("#units-input option:selected").text());
    });

    $.get("/settings/steps-per-motor-rotation", (async = false), (value) => {
        $("#motor-steps-per-rotation-input").val(parseFloat(value))
    });

    $.get("/settings/distance-per-motor-rotation", (async = false), (value) => {
        $("#distance-per-rotation-input").val(parseFloat(value))
    });
});
