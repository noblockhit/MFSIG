$(window).on("load", function () {
    $.get("/cameras", function (data) {
        $("#cam-select").html(
            `<option value="-1" selected disabled hidden>No camera selected...</option>` +
                data
        );
    });

    $("#cam-select").on("change", function () {
        $.ajax({
            url: "/camera/"+this.value,
            type: "POST",
            dataType: "text",
            success: function (data) {
                console.log("success "+ data)
                $("#reso-select").html(
                    `<option value="-1" selected disabled hidden>No resolution selected...</option>` +
                        data
                );
            },
            
            error: function (err) {
                console.log("Ajax error:")
                console.log(err);
            }
        });
    });

    $("#reso-select").on("change", function () {
        $.ajax({
            url: "/resolution/"+this.value,
            type: "POST",         
            error: function (err) {
                console.log("Ajax error:")
                console.log(err);
            }
        });
    });
});
