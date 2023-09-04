function prevent_submit_and_unfocus(e) {
    document.activeElement.blur()
    e.preventDefault();
}

$(window).on("load", () => {
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
});

console.log("here")