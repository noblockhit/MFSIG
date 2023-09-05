function prevent_submit_and_unfocus(e) {
    document.activeElement.blur()
    e.preventDefault();
}

$(window).on("load", () => {
    const mspr_input = document.getElementById("motor-steps-per-rotation-input");
    const dpr_input = document.getElementById("distance-per-rotation-input");

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
        $.post(`/settings/steps-per-motor-rotation/${mspr_input.value}`);
    });

    $(dpr_input).on("keyup", function (e) {
        $.post(`/settings/distance-per-rotation-input/${dpr_input.value}`);
    });

    // load saved values

    $.get("/settings/GPIO-default-on", (async=false), (value) => {
        $("#gpio-default-on-input").val(value);
    });

    $.get("/settings/GPIO-motor-pins", (async=false), (value) => {
        console.log(value)
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
